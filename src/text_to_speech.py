from google.cloud import texttospeech
import logging
import re
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
import os


class TextToSpeech:
    def __init__(self, language='ar', use_google=False):
        """
        Initialize the TextToSpeech class.

        Parameters:
        - language: Language code (default 'ar' for Arabic).
        - use_google: If True, use Google Cloud Text-to-Speech API. Defaults to False (uses gTTS).
        """
        self.language = language
        self.use_google = use_google

    def _preprocess_text(self, text):
        """
        Preprocess text to improve speech continuity:
        1. Remove extra whitespaces.
        2. Replace multiple newlines with a single space.
        3. Remove unnecessary punctuation that might cause pauses.
        """
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        text = text.replace('؛', ',')    # Replace Arabic semicolon with comma
        text = text.replace('»', '').replace('«', '')  # Remove quotes
        text = re.sub(r'\n\n', '. ', text).replace('\n', ' ')  # Normalize newlines
        return text.strip()

    def _google_tts(self, text, output_path):
        """
        Use Google Cloud Text-to-Speech API for generating audio.

        Parameters:
        - text: Input text to convert to speech.
        - output_path: Path to save the audio file.
        """
        try:
            # Load environment variables from .env file
            load_dotenv()

            # Get the credentials path from the .env file
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

            if not credentials_path:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in the .env file.")

            # Set the environment variable for Google Cloud
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

            # Initialize Google Cloud TTS client
            client = texttospeech.TextToSpeechClient()

            # Set up the text input
            input_text = texttospeech.SynthesisInput(text=text)

            # Configure voice
            voice = texttospeech.VoiceSelectionParams(
                language_code="ar-XA",  # Replace with preferred dialect
                ssml_gender=texttospeech.SsmlVoiceGender.MALE,
            )

            # Configure audio output
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

            # Generate the speech
            response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

            # Save the audio file
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
            print(f"Audio saved to {output_path} using Google Cloud TTS.")
        except Exception as e:
            logging.error(f"Google Cloud TTS failed: {e}")

    def text_to_audio(self, text, output_path, chunk_size=4500):
        """
        Convert text to audio, handling text longer than Google TTS limits.

        Parameters:
        - text: Input text to convert to speech.
        - output_path: Path to save the combined audio file.
        - chunk_size: Maximum byte size for each TTS request (default: 4500).
        """
        try:
            # Preprocess and chunk the text
            chunks = []
            current_chunk = ""

            for word in text.split():
                if len((current_chunk + " " + word).encode("utf-8")) > chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    current_chunk += " " + word

            if current_chunk:
                chunks.append(current_chunk.strip())

            logging.info(f"Split text into {len(chunks)} chunks for processing.")

            # Process each chunk and save audio
            audio_segments = []
            for i, chunk in enumerate(chunks):
                chunk_path = f"{output_path.rsplit('.', 1)[0]}_{i}.mp3"
                self._google_tts(chunk, chunk_path)
                audio_segments.append(AudioSegment.from_mp3(chunk_path))

            # Combine all audio segments into one file
            combined = sum(audio_segments)
            combined.export(output_path, format="mp3")

            # Clean up temporary chunk files
            for i in range(len(chunks)):
                chunk_path = f"{output_path.rsplit('.', 1)[0]}_{i}.mp3"
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)

            logging.info(f"Combined audio saved to {output_path}")

        except Exception as e:
            logging.error(f"Failed to generate audio: {e}")
            raise
