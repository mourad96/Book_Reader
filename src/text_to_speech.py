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
        - use_google: If True, use Google Cloud Text-to-Speech API. Defaults to False (uses gTTS offline).
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
                language_code="ar-XA",
                ssml_gender=texttospeech.SsmlVoiceGender.MALE,
            )

            # Configure audio output
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

            # Generate the speech
            response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

            # Save the audio file
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
            logging.info(f"Audio saved to {output_path} using Google Cloud TTS.")
        except Exception as e:
            logging.error(f"Google Cloud TTS failed: {e}")
            raise

    def _gtts_offline(self, text, output_path, chunk_size=4500):
        """
        Use gTTS offline to convert text to speech.

        Parameters:
        - text: Input text to convert to speech.
        - output_path: Path to save the audio file.
        - chunk_size: Maximum characters per request (default: 4500).
        """
        try:
            # Preprocess and chunk the text
            chunks = []
            processed_text = self._preprocess_text(text)

            while processed_text:
                if len(processed_text) <= chunk_size:
                    chunks.append(processed_text)
                    break
                split_point = processed_text.rfind(' ', 0, chunk_size)
                if split_point == -1:
                    split_point = chunk_size
                chunks.append(processed_text[:split_point])
                processed_text = processed_text[split_point:].strip()

            # Generate audio for each chunk
            if len(chunks) == 1:
                tts = gTTS(text=chunks[0], lang=self.language, slow=False)
                tts.save(output_path)
            else:
                audio_segments = []
                for i, chunk in enumerate(chunks):
                    chunk_path = f"{output_path.rsplit('.', 1)[0]}_{i}.mp3"
                    tts = gTTS(text=chunk, lang=self.language, slow=False)
                    tts.save(chunk_path)
                    audio_segments.append(AudioSegment.from_mp3(chunk_path))

                # Combine all audio segments into one file
                combined = sum(audio_segments)
                combined.export(output_path, format="mp3")

                # Clean up temporary chunk files
                for i in range(len(chunks)):
                    os.remove(f"{output_path.rsplit('.', 1)[0]}_{i}.mp3")

            logging.info(f"Audio saved to {output_path} using gTTS offline.")
        except Exception as e:
            logging.error(f"gTTS offline TTS failed: {e}")
            raise

    def text_to_audio(self, text, output_path, chunk_size=4500):
        """
        Convert text to speech using the selected TTS method (Google Cloud or gTTS offline).

        Parameters:
        - text: Input text to convert to speech.
        - output_path: Path to save the combined audio file.
        - chunk_size: Maximum byte size for each request (default: 4500).
        """
        if self.use_google:
            self._google_tts(text, output_path)
        else:
            self._gtts_offline(text, output_path, chunk_size=chunk_size)
