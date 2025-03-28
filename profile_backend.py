import cProfile
import pstats
import uvicorn
from app.main import app
import logging
from time import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_server():
    # Start profiling
    pr = cProfile.Profile()
    pr.enable()

    # Log server start
    logger.info("Starting FastAPI server...")

    # Run your FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=5001)

    # Stop profiling
    pr.disable()
    with open("backend_profile.prof", "w") as f:
        ps = pstats.Stats(pr, stream=f)
        ps.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats()

if __name__ == "__main__":
    run_server() 