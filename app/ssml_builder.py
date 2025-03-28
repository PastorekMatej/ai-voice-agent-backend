import html
import re

class SSMLBuilder:
    """Classe pour construire des documents SSML de manière propre et modulaire."""
    
    def __init__(self, text=None):
        self.elements = []
        if text:
            self.add_text(text)
    
    def add_text(self, text):
        """Ajoute du texte en échappant les caractères spéciaux XML."""
        self.elements.append(html.escape(text))
        return self
    
    def add_break(self, time=None, strength=None):
        """Ajoute une pause."""
        if time:
            self.elements.append(f'<break time="{time}"/>')
        elif strength:
            self.elements.append(f'<break strength="{strength}"/>')
        return self
    
    def add_emphasis(self, text, level="moderate"):
        """Ajoute du texte avec emphase."""
        self.elements.append(f'<emphasis level="{level}">{html.escape(text)}</emphasis>')
        return self
    
    def add_prosody(self, text, rate=None, pitch=None, volume=None):
        """Ajoute du texte avec prosodie modifiée."""
        attrs = []
        if rate: attrs.append(f'rate="{rate}"')
        if pitch: attrs.append(f'pitch="{pitch}"')
        if volume: attrs.append(f'volume="{volume}"')
        
        attrs_str = " ".join(attrs)
        self.elements.append(f'<prosody {attrs_str}>{html.escape(text)}</prosody>')
        return self
    
    def add_say_as(self, text, interpret_as, format=None):
        """Ajoute du texte avec interprétation spécifique."""
        format_attr = f' format="{format}"' if format else ''
        self.elements.append(f'<say-as interpret-as="{interpret_as}"{format_attr}>{html.escape(text)}</say-as>')
        return self
    
    def add_paragraph(self, text):
        """Ajoute un paragraphe."""
        self.elements.append(f'<p>{html.escape(text)}</p>')
        return self
    
    def add_sentence(self, text):
        """Ajoute une phrase."""
        self.elements.append(f'<s>{html.escape(text)}</s>')
        return self
    
    def to_ssml(self):
        """Convertit les éléments en document SSML complet."""
        content = "".join(self.elements)
        return f"<speak>{content}</speak>"
    
    def apply_automatic_pauses(self, text):
        """Ajoute automatiquement des pauses basées sur la ponctuation."""
        result = html.escape(text)
        
        # Remplacer les ponctuations par des pauses
        punctuation_pauses = {
            '. ': '<break time="750ms"/> ',
            '! ': '<break time="750ms"/> ',
            '? ': '<break time="750ms"/> ',
            ', ': '<break time="400ms"/> ',
            '; ': '<break time="500ms"/> ',
            ': ': '<break time="500ms"/> '
        }
        
        for punct, pause in punctuation_pauses.items():
            result = result.replace(punct, punct[0] + pause)
        
        self.elements.append(result)
        return self
    
    def apply_automatic_emphasis(self, text):
        """Applique automatiquement l'emphase sur le texte selon des règles."""
        escaped_text = html.escape(text)
        
        # Mots entre *astérisques*
        result = re.sub(r'\*(.*?)\*', r'<emphasis level="moderate">\1</emphasis>', escaped_text)
        
        # Mots en MAJUSCULES
        result = re.sub(r'\b([A-Z]{3,})\b', r'<emphasis level="strong">\1</emphasis>', result)
        
        self.elements.append(result)
        return self 