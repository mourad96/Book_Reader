from google.cloud import texttospeech
import logging
import re
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
import os
import struct
import mimetypes
# Import Gemini SDK
from google import genai
from google.genai import types
import wave
import concurrent.futures
import threading
from collections import defaultdict


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
        text = text.replace('؛', ',')    # Replace Arabic semicolon with a comma
        text = text.replace('»', '').replace('«', '')  # Remove quotes
        text = re.sub(r'\n\n', '. ', text).replace('\n', ' ')  # Normalize newlines
        return text.strip()

    def _convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Generates a WAV file header for the given audio data and parameters.

        Args:
            audio_data: The raw audio data as a bytes object.
            mime_type: Mime type of the audio data.

        Returns:
            A bytes object representing the WAV file header.
        """
        parameters = self._parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1
        data_size = len(audio_data)
        bytes_per_sample = bits_per_sample // 8
        block_align = num_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

        # http://soundfile.sapp.org/doc/WaveFormat/
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",          # ChunkID
            chunk_size,       # ChunkSize (total file size - 8 bytes)
            b"WAVE",          # Format
            b"fmt ",          # Subchunk1ID
            16,               # Subchunk1Size (16 for PCM)
            1,                # AudioFormat (1 for PCM)
            num_channels,     # NumChannels
            sample_rate,      # SampleRate
            byte_rate,        # ByteRate
            block_align,      # BlockAlign
            bits_per_sample,  # BitsPerSample
            b"data",          # Subchunk2ID
            data_size         # Subchunk2Size (size of audio data)
        )
        return header + audio_data

    def _parse_audio_mime_type(self, mime_type: str) -> dict:
        """Parses bits per sample and rate from an audio MIME type string.

        Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

        Args:
            mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

        Returns:
            A dictionary with "bits_per_sample" and "rate" keys. Values will be
            integers if found, otherwise None.
        """
        bits_per_sample = 16
        rate = 24000

        # Extract rate from parameters
        parts = mime_type.split(";")
        for param in parts: # Skip the main type part
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    # Handle cases like "rate=" with no value or non-integer value
                    pass # Keep rate as default
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass # Keep bits_per_sample as default if conversion fails

        return {"bits_per_sample": bits_per_sample, "rate": rate}

    def _save_binary_file(self, file_name, data):
        """Save binary data to file."""
        with open(file_name, "wb") as f:
            f.write(data)
        logging.info(f"File saved to: {file_name}")



    def _google_tts(self, text, output_path, chunk_size=5000,
                model="gemini-2.5-flash-preview-tts", voice_name="Charon", style_instructions="Read with a natural storytelling tone, clear enunciation, and moderate pace. Emphasize sentence rhythm and end with subtle intonation shifts at periods and commas.", max_workers=5):
        """
        Use Google Gemini Flash TTS (via the Generative AI SDK) for generating audio with parallel processing.

        Parameters:
        - text:       Input text to convert to speech.
        - output_path:Path to save the final MP3 file.
        - chunk_size: Maximum UTF-8 byte size per TTS request.
        - model:      Gemini TTS model to use.
        - style_instructions: Optional style instructions for speech generation (e.g., "Speak slowly and clearly", "Use an enthusiastic tone")
        - max_workers: Maximum number of parallel workers (default: 5)
        """
        try:
            # Load .env and ensure we have an API key
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set in .env")
            
            client = genai.Client(api_key=api_key)

            # Preprocess and chunk text by byte size
            processed = text
            
            # Add style instructions to the text if provided
            if style_instructions:
                processed = f"{style_instructions} {processed}"
            
            chunks, cur = [], ""
            for word in processed.split():
                if len((cur + " " + word).encode("utf-8")) > chunk_size:
                    if cur:
                        chunks.append(cur.strip())
                    cur = word
                else:
                    cur += " " + word if cur else word
            if cur:
                chunks.append(cur.strip())

            logging.info(f"Splitting into {len(chunks)} chunks for Gemini TTS.")

            def process_chunk(chunk_data):
                """Process a single chunk and return (index, audio_data, mime_type)"""
                chunk_index, chunk_text = chunk_data
                
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=chunk_text)],
                    ),
                ]
                
                # Build speech config
                generate_content_config = types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name
                            )
                        )
                    ),
                )

                # Generate audio using streaming API
                audio_data = b""
                mime_type = None
                
                try:
                    for packet in client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=generate_content_config,
                    ):
                        if (
                            packet.candidates is None
                            or packet.candidates[0].content is None
                            or packet.candidates[0].content.parts is None
                        ):
                            continue
                        
                        if (packet.candidates[0].content.parts[0].inline_data and 
                            packet.candidates[0].content.parts[0].inline_data.data):
                            inline_data = packet.candidates[0].content.parts[0].inline_data
                            audio_data += inline_data.data
                            if mime_type is None:
                                mime_type = inline_data.mime_type

                    return chunk_index, audio_data, mime_type
                    
                except Exception as e:
                    logging.error(f"Error processing chunk {chunk_index}: {e}")
                    return chunk_index, None, None

            # Process chunks in parallel using ThreadPoolExecutor
            chunk_results = {}  # Dictionary to store results by index
            base, _ = os.path.splitext(output_path)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunks for processing
                future_to_index = {
                    executor.submit(process_chunk, (i, chunk)): i 
                    for i, chunk in enumerate(chunks)
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_index):
                    chunk_index = future_to_index[future]
                    try:
                        index, audio_data, mime_type = future.result()
                        if audio_data:
                            chunk_results[index] = (audio_data, mime_type)
                            logging.info(f"Completed chunk {index + 1}/{len(chunks)}")
                        else:
                            logging.error(f"No audio data for chunk {index}")
                            chunk_results[index] = None
                    except Exception as e:
                        logging.error(f"Chunk {chunk_index} generated an exception: {e}")
                        chunk_results[chunk_index] = None

            # Process results in correct order and create audio segments
            audio_segments = []
            for i in range(len(chunks)):
                if i in chunk_results and chunk_results[i] is not None:
                    audio_data, mime_type = chunk_results[i]
                    
                    # Convert to WAV format
                    file_extension = mimetypes.guess_extension(mime_type)
                    if file_extension is None:
                        file_extension = ".wav"
                        wav_data = self._convert_to_wav(audio_data, mime_type)
                    else:
                        wav_data = audio_data
                    
                    # Save temporary WAV file
                    tmp_wav = f"{base}_{i}.wav"
                    self._save_binary_file(tmp_wav, wav_data)
                    audio_segments.append(AudioSegment.from_wav(tmp_wav))
                else:
                    logging.warning(f"Chunk {i} failed to process, skipping...")

            # Concatenate and export as MP3
            if audio_segments:
                combined = sum(audio_segments)
                combined.export(output_path, format="mp3")

                # Cleanup temporary files
                for i in range(len(chunks)):
                    tmp_file = f"{base}_{i}.wav"
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)

                logging.info(f"Combined audio saved to {output_path} using Gemini Flash TTS (parallel processing).")
            else:
                logging.error("No audio data generated from any chunks")
                raise RuntimeError("No audio data generated from any chunks")

        except Exception as e:
            logging.error(f"Gemini Flash TTS failed: {e}")
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

    def text_to_audio(self, text, output_path, chunk_size=5000, style_instructions="Read aloud in a warm and friendly tone:"):
        """
        Convert text to speech using the selected TTS method (Google Cloud or gTTS offline).

        Parameters:
        - text: Input text to convert to speech.
        - output_path: Path to save the combined audio file.
        - chunk_size: Maximum byte size for each request (default: 5000 for Google TTS).
        - style_instructions: Optional style instructions for speech generation (only for Google TTS).
        """
        if self.use_google:
            self._google_tts(text, output_path, chunk_size=chunk_size, style_instructions=style_instructions)
        else:
            self._gtts_offline(text, output_path, chunk_size=chunk_size)