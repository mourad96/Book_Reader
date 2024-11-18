import logging
from gtts import gTTS

class TextToSpeech:
    def __init__(self, language='ar'):
        self.language = language

    def text_to_audio(self, text, output_path):
        """Convert text to speech and save as an audio file."""
        try:
            tts = gTTS(text=text, lang=self.language, slow=False)
            tts.save(output_path)
            print(f"Audio saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to generate audio: {e}")