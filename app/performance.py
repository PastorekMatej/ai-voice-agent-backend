import time
import logging
import json
import os
from pathlib import Path
import threading
from datetime import datetime
from collections import defaultdict
from functools import wraps
from statistics import mean, median, stdev
import tempfile
import subprocess
import io
from pydantic import BaseModel

# For benchmarking
import sys
import requests
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
import argparse

# Configure logger
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Performance monitoring system for the backend application.
    Tracks detailed metrics for modules and functions execution.
    """
    
    def __init__(self, export_path=None):
        self.metrics = defaultdict(list)
        self.export_path = export_path or Path(__file__).parent.parent / 'metrics'
        self._lock = threading.Lock()
        
        # Create metrics directory if it doesn't exist
        os.makedirs(self.export_path, exist_ok=True)
        
        # Configure periodic export 
        self.export_interval = 3600  # 1 hour by default
        self._setup_periodic_export()
    
    def record_metric(self, module, function, duration, metadata=None):
        """Record a performance metric."""
        with self._lock:
            timestamp = datetime.now().isoformat()
            self.metrics[f"{module}.{function}"].append({
                'timestamp': timestamp,
                'duration': duration,
                'metadata': metadata or {}
            })
            
            # Write to text file
            self._write_to_text_file(module, function, timestamp, duration, metadata)
    
    def _write_to_text_file(self, module, function, timestamp, duration, metadata):
        """Write a single metric to the text file."""
        filepath = Path(self.export_path) / "performance_metrics.txt"
        
        with open(filepath, 'a') as f:
            metadata_str = "" if not metadata else f", metadata: {json.dumps(metadata)}"
            f.write(f"{timestamp} - {module}.{function}: {duration:.2f}ms{metadata_str}\n")
    
    def get_metrics(self, module=None, function=None, limit=None):
        """Get recorded metrics, optionally filtered by module and function."""
        with self._lock:
            result = {}
            
            for key, measurements in self.metrics.items():
                mod, func = key.split('.')
                
                if module and mod != module:
                    continue
                if function and func != function:
                    continue
                
                result[key] = measurements[-limit:] if limit else measurements.copy()
            
            return result
    
    def get_summary(self, module=None, function=None):
        """Get statistical summary of metrics."""
        filtered_metrics = self.get_metrics(module, function)
        summary = {}
        
        for key, measurements in filtered_metrics.items():
            if not measurements:
                continue
                
            durations = [m['duration'] for m in measurements]
            
            try:
                summary[key] = {
                    'count': len(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'mean': mean(durations),
                    'median': median(durations),
                    'stdev': stdev(durations) if len(durations) > 1 else 0,
                    'total': sum(durations),
                    'last_execution': measurements[-1]['timestamp']
                }
            except Exception as e:
                logger.error(f"Error calculating statistics for {key}: {str(e)}")
                summary[key] = {'error': str(e), 'count': len(durations)}
        
        return summary
    
    def reset_metrics(self, module=None, function=None):
        """Reset metrics, optionally for a specific module/function."""
        with self._lock:
            if module and function:
                key = f"{module}.{function}"
                if key in self.metrics:
                    self.metrics[key] = []
            elif module:
                for key in list(self.metrics.keys()):
                    if key.startswith(f"{module}."):
                        self.metrics[key] = []
            else:
                self.metrics = defaultdict(list)
    
    def _setup_periodic_export(self):
        """Setup periodic export of metrics summary to text file."""
        def export_task():
            self._export_summary_to_text()
            threading.Timer(self.export_interval, export_task).start()
        
        # Start the first timer
        threading.Timer(self.export_interval, export_task).start()
    
    def _export_summary_to_text(self):
        """Export summary metrics to a text file."""
        summary = self.get_summary()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Path(self.export_path) / f"metrics_summary_{timestamp}.txt"
        
        with open(filepath, 'w') as f:
            f.write(f"Performance Metrics Summary - {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            for key, data in summary.items():
                f.write(f"Module.Function: {key}\n")
                f.write("-" * 50 + "\n")
                for stat_name, stat_value in data.items():
                    if isinstance(stat_value, float):
                        f.write(f"  {stat_name}: {stat_value:.2f}\n")
                    else:
                        f.write(f"  {stat_name}: {stat_value}\n")
                f.write("\n")


# Create a global instance of the performance monitor
monitor = PerformanceMonitor()


def measure_performance(module=None):
    """
    Decorator to measure function execution time.
    
    Usage:
        @measure_performance("module_name")
        def my_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine module name
            mod_name = module or func.__module__.split('.')[-1]
            
            # Start timing
            start_time = time.perf_counter()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                return result
            finally:
                # Calculate duration and record metric
                duration = (time.perf_counter() - start_time) * 1000  # Convert to ms
                monitor.record_metric(mod_name, func.__name__, duration)
                logger.debug(f"Performance: {mod_name}.{func.__name__} took {duration:.2f}ms")
        
        return wrapper
    return decorator


# Context manager for measuring performance of code blocks
class PerformanceTracker:
    """
    Context manager to measure performance of code blocks.
    
    Usage:
        with PerformanceTracker("module", "operation"):
            # code to measure
    """
    def __init__(self, module, operation, metadata=None):
        self.module = module
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.perf_counter() - self.start_time) * 1000  # Convert to ms
        monitor.record_metric(self.module, self.operation, duration, self.metadata)
        logger.debug(f"Performance: {self.module}.{self.operation} took {duration:.2f}ms")


# ========= Benchmark Functions =========

class BenchmarkResults:
    """Stores and analyzes benchmark results"""
    
    def __init__(self):
        self.results = defaultdict(list)
    
    def add_result(self, operation, duration):
        self.results[operation].append(duration)
    
    def get_summary(self):
        summary = {}
        for operation, durations in self.results.items():
            summary[operation] = {
                'count': len(durations),
                'min': min(durations),
                'max': max(durations),
                'mean': mean(durations),
                'median': median(durations),
                'stdev': stdev(durations) if len(durations) > 1 else 0,
                'total': sum(durations)
            }
        return summary
    
    def export_to_file(self, filepath):
        with open(filepath, 'w') as f:
            f.write(f"Benchmark Results - {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            summary = self.get_summary()
            for operation, stats in summary.items():
                f.write(f"Operation: {operation}\n")
                f.write("-" * 50 + "\n")
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, float):
                        f.write(f"  {stat_name}: {stat_value:.2f}ms\n")
                    else:
                        f.write(f"  {stat_name}: {stat_value}\n")
                f.write("\n")
            
            f.write("\nRaw Data:\n")
            f.write("-" * 50 + "\n")
            for operation, durations in self.results.items():
                f.write(f"{operation}: {', '.join([f'{d:.2f}ms' for d in durations])}\n")


def benchmark_speech_to_text(audio_file_path, num_runs=3):
    """Benchmark speech-to-text performance with a standard audio file"""
    results = BenchmarkResults()
    
    for i in range(num_runs):
        print(f"STT Benchmark Run {i+1}/{num_runs}...")
        
        # File reading
        start_time = time.perf_counter()
        with open(audio_file_path, 'rb') as audio_file:
            audio_content = audio_file.read()
        file_read_time = (time.perf_counter() - start_time) * 1000
        results.add_result("stt.file_reading", file_read_time)
        
        # Audio conversion (if needed)
        if audio_file_path.endswith('.webm'):
            start_time = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                wav_path = wav_file.name
            
            ffmpeg_cmd = [
                r"C:\Users\pasto\ffmpeg\ffmpeg.exe",
                '-y',
                '-i', audio_file_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-f', 'wav',
                wav_path
            ]
            
            process = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                shell=True
            )
            
            with open(wav_path, 'rb') as wav_file:
                audio_content = wav_file.read()
            
            os.unlink(wav_path)
            conversion_time = (time.perf_counter() - start_time) * 1000
            results.add_result("stt.audio_conversion", conversion_time)
        
        # Client initialization
        start_time = time.perf_counter()
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="fr-FR",
        )
        init_time = (time.perf_counter() - start_time) * 1000
        results.add_result("stt.client_initialization", init_time)
        
        # API call
        start_time = time.perf_counter()
        response = client.recognize(config=config, audio=audio)
        api_time = (time.perf_counter() - start_time) * 1000
        results.add_result("stt.google_api_call", api_time)
        
        # Text extraction
        start_time = time.perf_counter()
        transcription = ""
        for result in response.results:
            transcription += result.alternatives[0].transcript
        extraction_time = (time.perf_counter() - start_time) * 1000
        results.add_result("stt.text_extraction", extraction_time)
        
        # Total time
        total_time = file_read_time + (conversion_time if audio_file_path.endswith('.webm') else 0) + init_time + api_time + extraction_time
        results.add_result("stt.total", total_time)
    
    return results


def benchmark_text_to_speech(text, num_runs=3):
    """Benchmark text-to-speech performance with a standard text"""
    results = BenchmarkResults()
    
    # Import the SSML builder here to avoid circular imports
    sys.path.append(str(Path(__file__).parent))
    from ssml_builder import SSMLBuilder
    
    for i in range(num_runs):
        print(f"TTS Benchmark Run {i+1}/{num_runs}...")
        
        # Client initialization
        start_time = time.perf_counter()
        client = texttospeech.TextToSpeechClient()
        init_time = (time.perf_counter() - start_time) * 1000
        results.add_result("tts.client_initialization", init_time)
        
        # SSML generation
        start_time = time.perf_counter()
        # Simple SSML generation for benchmarking
        builder = SSMLBuilder()
        builder.add_text(text)
        ssml_text = builder.to_ssml()
        input_text = texttospeech.SynthesisInput(ssml=ssml_text)
        ssml_time = (time.perf_counter() - start_time) * 1000
        results.add_result("tts.ssml_generation", ssml_time)
        
        # Voice configuration
        start_time = time.perf_counter()
        voice = texttospeech.VoiceSelectionParams(
            language_code="fr-FR",
            name="fr-FR-Neural2-D",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0
        )
        config_time = (time.perf_counter() - start_time) * 1000
        results.add_result("tts.voice_configuration", config_time)
        
        # API call
        start_time = time.perf_counter()
        response = client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        api_time = (time.perf_counter() - start_time) * 1000
        results.add_result("tts.google_api_call", api_time)
        
        # File writing (to temporary file)
        start_time = time.perf_counter()
        logger.debug(f"type(response.audio_content): {type(response.audio_content)}")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as out:
            out.write(response.audio_content)
        file_write_time = (time.perf_counter() - start_time) * 1000
        results.add_result("tts.file_writing", file_write_time)
        
        # Total time
        total_time = init_time + ssml_time + config_time + api_time + file_write_time
        results.add_result("tts.total", total_time)
    
    return results


def benchmark_chat_completion(message, num_runs=3):
    """Benchmark chat completion performance with a standard message"""
    results = BenchmarkResults()
    
    # Import the OpenAI client
    sys.path.append(str(Path(__file__).parent.parent))
    from dotenv import load_dotenv
    from openai import OpenAI
    
    # Load environment variables
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    
    for i in range(num_runs):
        print(f"Chat Completion Benchmark Run {i+1}/{num_runs}...")
        
        # Client initialization
        start_time = time.perf_counter()
        client = OpenAI(api_key=openai_key)
        init_time = (time.perf_counter() - start_time) * 1000
        results.add_result("chat.client_initialization", init_time)
        
        # Message preparation
        start_time = time.perf_counter()
        messages = [
            {"role": "system", "content": "Tu es un assistant qui aide à apprendre le français."},
            {"role": "user", "content": message}
        ]
        prep_time = (time.perf_counter() - start_time) * 1000
        results.add_result("chat.message_preparation", prep_time)
        
        # API call
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            top_p=1.0
        )
        api_time = (time.perf_counter() - start_time) * 1000
        results.add_result("chat.openai_api_call", api_time)
        
        # Response processing
        start_time = time.perf_counter()
        ai_text = response.choices[0].message.content
        process_time = (time.perf_counter() - start_time) * 1000
        results.add_result("chat.response_processing", process_time)
        
        # Total time
        total_time = init_time + prep_time + api_time + process_time
        results.add_result("chat.total", total_time)
    
    return results


def run_benchmarks(audio_file=None, text=None, message=None, num_runs=3):
    """Run all benchmarks with standardized inputs"""
    results_dir = Path(__file__).parent.parent / 'metrics' / 'benchmarks'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Default values if not provided
    audio_file = audio_file or str(Path(__file__).parent.parent / 'samples' / 'test_audio.wav')
    text = text or "Bonjour, je suis un assistant virtuel qui vous aide à apprendre le français. Comment puis-je vous aider aujourd'hui?"
    message = message or "Comment conjuguer le verbe être au présent?"
    
    # Ensure the sample directory exists
    sample_dir = Path(__file__).parent.parent / 'samples'
    os.makedirs(sample_dir, exist_ok=True)
    
    # Check if audio file exists, otherwise print warning
    if not os.path.exists(audio_file):
        print(f"Warning: Audio file {audio_file} does not exist. STT benchmark will be skipped.")
    
    # Run benchmarks
    all_results = {}
    
    if os.path.exists(audio_file):
        print(f"\nRunning Speech-to-Text benchmark with: {audio_file}")
        stt_results = benchmark_speech_to_text(audio_file, num_runs)
        stt_file = results_dir / f"stt_benchmark_{timestamp}.txt"
        stt_results.export_to_file(stt_file)
        print(f"STT benchmark results saved to: {stt_file}")
        all_results["stt"] = stt_results.get_summary()
    
    print(f"\nRunning Text-to-Speech benchmark with: {text}")
    tts_results = benchmark_text_to_speech(text, num_runs)
    tts_file = results_dir / f"tts_benchmark_{timestamp}.txt"
    tts_results.export_to_file(tts_file)
    print(f"TTS benchmark results saved to: {tts_file}")
    all_results["tts"] = tts_results.get_summary()
    
    print(f"\nRunning Chat benchmark with: {message}")
    chat_results = benchmark_chat_completion(message, num_runs)
    chat_file = results_dir / f"chat_benchmark_{timestamp}.txt"
    chat_results.export_to_file(chat_file)
    print(f"Chat benchmark results saved to: {chat_file}")
    all_results["chat"] = chat_results.get_summary()
    
    # Save combined results
    combined_file = results_dir / f"combined_benchmark_{timestamp}.txt"
    with open(combined_file, 'w') as f:
        f.write(f"Combined Benchmark Results - {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        for module, results in all_results.items():
            f.write(f"Module: {module.upper()}\n")
            f.write("=" * 50 + "\n")
            
            for operation, stats in results.items():
                f.write(f"Operation: {operation}\n")
                f.write("-" * 40 + "\n")
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, float):
                        f.write(f"  {stat_name}: {stat_value:.2f}ms\n")
                    else:
                        f.write(f"  {stat_name}: {stat_value}\n")
                f.write("\n")
            f.write("\n")
    
    print(f"\nAll benchmark results combined in: {combined_file}")
    print("\nBenchmark Summary:")
    print("=" * 50)
    
    # Print summary of means
    for module, results in all_results.items():
        print(f"\n{module.upper()} Operations:")
        for operation, stats in results.items():
            if operation.endswith('total'):
                print(f"  Total: {stats['mean']:.2f}ms (min: {stats['min']:.2f}ms, max: {stats['max']:.2f}ms)")
    
    return all_results


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Run performance benchmarks for the AI Voice Agent")
    parser.add_argument("--audio", help="Path to audio file for speech-to-text benchmark", default=None)
    parser.add_argument("--text", help="Text to use for text-to-speech benchmark", default=None)
    parser.add_argument("--message", help="Message to use for chat completion benchmark", default=None)
    parser.add_argument("--runs", type=int, help="Number of benchmark runs to perform", default=3)
    
    args = parser.parse_args()
    
    # Run benchmarks
    run_benchmarks(args.audio, args.text, args.message, args.runs) 