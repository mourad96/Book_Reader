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
import time


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


    def _smart_text_chunking(self, text, max_byte_size=5000):
        """
        Improved text chunking that respects sentence boundaries and avoids word splitting.
        
        Parameters:
        - text: Input text to chunk
        - max_byte_size: Maximum byte size per chunk
        
        Returns:
        - List of text chunks
        """
        # Preprocess the text first
        processed_text = text
        
        # Split into sentences first to respect natural boundaries
        sentence_endings = re.compile(r'[.!?؟][\s]+')
        sentences = sentence_endings.split(processed_text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed the byte limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk.encode('utf-8')) <= max_byte_size:
                current_chunk = test_chunk
            else:
                # If current_chunk is not empty, save it and start new chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # If single sentence is too long, split by words
                    words = sentence.split()
                    temp_chunk = ""
                    
                    for word in words:
                        test_word_chunk = temp_chunk + " " + word if temp_chunk else word
                        
                        if len(test_word_chunk.encode('utf-8')) <= max_byte_size:
                            temp_chunk = test_word_chunk
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word
                    
                    current_chunk = temp_chunk
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Remove empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks

    def _convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Generates a WAV file header for the given audio data and parameters."""
        parameters = self._parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1
        data_size = len(audio_data)
        bytes_per_sample = bits_per_sample // 8
        block_align = num_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        chunk_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1,
            num_channels, sample_rate, byte_rate, block_align,
            bits_per_sample, b"data", data_size
        )
        return header + audio_data

    def _parse_audio_mime_type(self, mime_type: str) -> dict:
        """Parses bits per sample and rate from an audio MIME type string."""
        bits_per_sample = 16
        rate = 24000

        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    pass
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass

        return {"bits_per_sample": bits_per_sample, "rate": rate}

    def _save_binary_file(self, file_name, data):
        """Save binary data to file."""
        with open(file_name, "wb") as f:
            f.write(data)
        logging.info(f"File saved to: {file_name}")

    def _process_single_chunk(self, chunk_index, chunk_text, model, voice_name, client):
        """
        Process a single chunk and return the result.
        
        Returns:
        - tuple: (chunk_index, success, audio_data, mime_type, error_message)
        """
        try:
            logging.info(f"Processing chunk {chunk_index + 1}: '{chunk_text[:50]}...'")
            
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=chunk_text)],
                ),
            ]
            
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

            audio_data = b""
            mime_type = None
            
            for packet in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if (packet.candidates is None or 
                    packet.candidates[0].content is None or 
                    packet.candidates[0].content.parts is None):
                    continue
                
                if (packet.candidates[0].content.parts[0].inline_data and 
                    packet.candidates[0].content.parts[0].inline_data.data):
                    inline_data = packet.candidates[0].content.parts[0].inline_data
                    audio_data += inline_data.data
                    if mime_type is None:
                        mime_type = inline_data.mime_type

            if not audio_data:
                return chunk_index, False, None, None, "No audio data received"
            
            logging.info(f"Successfully processed chunk {chunk_index + 1} ({len(audio_data)} bytes)")
            return chunk_index, True, audio_data, mime_type, None
                    
        except Exception as e:
            error_msg = f"Error processing chunk {chunk_index}: {str(e)}"
            logging.error(error_msg)
            return chunk_index, False, None, None, error_msg

    def _google_tts(self, text, output_path, chunk_size=5000,
                model="gemini-2.5-flash-preview-tts", voice_name="Charon", 
                style_instructions="Read with a natural storytelling tone, clear enunciation, and moderate pace. Emphasize sentence rhythm and end with subtle intonation shifts at periods and commas.", 
                max_workers=3, max_retries=2):
        """
        Use Google Gemini Flash TTS with improved error handling and chunk management.
        """
        try:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set in .env")
            
            client = genai.Client(api_key=api_key)

            # Smart chunking that respects sentence boundaries
            chunks = self._smart_text_chunking(text, chunk_size)
            
            # Add style instructions to first chunk only to avoid repetition
            if style_instructions and chunks:
                chunks[0] = f"{style_instructions} {chunks[0]}"
            
            logging.info(f"Split text into {len(chunks)} chunks for Gemini TTS.")
            for i, chunk in enumerate(chunks):
                logging.info(f"Chunk {i + 1} length: {len(chunk)} chars, {len(chunk.encode('utf-8'))} bytes")

            # Dictionary to store results with guaranteed ordering
            chunk_results = {}
            base, _ = os.path.splitext(output_path)
            failed_chunks = []
            
            # Process chunks with limited parallelism to avoid rate limits
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit chunks for processing
                future_to_index = {}
                for i, chunk in enumerate(chunks):
                    future = executor.submit(
                        self._process_single_chunk, i, chunk, model, voice_name, client
                    )
                    future_to_index[future] = i
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_index):
                    chunk_index, success, audio_data, mime_type, error_msg = future.result()
                    
                    if success:
                        chunk_results[chunk_index] = (audio_data, mime_type)
                        logging.info(f"✓ Chunk {chunk_index + 1}/{len(chunks)} completed successfully")
                    else:
                        logging.error(f"✗ Chunk {chunk_index + 1} failed: {error_msg}")
                        failed_chunks.append(chunk_index)

            # Retry failed chunks sequentially
            for retry_attempt in range(max_retries):
                if not failed_chunks:
                    break
                    
                logging.info(f"Retry attempt {retry_attempt + 1}/{max_retries} for {len(failed_chunks)} failed chunks")
                retry_failed = []
                
                for chunk_index in failed_chunks:
                    time.sleep(1)  # Rate limiting
                    chunk_index, success, audio_data, mime_type, error_msg = self._process_single_chunk(
                        chunk_index, chunks[chunk_index], model, voice_name, client
                    )
                    
                    if success:
                        chunk_results[chunk_index] = (audio_data, mime_type)
                        logging.info(f"✓ Retry successful for chunk {chunk_index + 1}")
                    else:
                        logging.error(f"✗ Retry failed for chunk {chunk_index + 1}: {error_msg}")
                        retry_failed.append(chunk_index)
                
                failed_chunks = retry_failed

            # Check if we have all chunks
            missing_chunks = [i for i in range(len(chunks)) if i not in chunk_results]
            if missing_chunks:
                raise RuntimeError(f"Failed to process chunks: {[i+1 for i in missing_chunks]}")

            # Process results in correct order
            audio_segments = []
            temp_files = []
            
            for i in range(len(chunks)):
                if i not in chunk_results:
                    logging.error(f"Missing chunk {i + 1}, cannot continue")
                    raise RuntimeError(f"Missing chunk {i + 1}")
                
                audio_data, mime_type = chunk_results[i]
                
                # Convert to WAV format
                file_extension = mimetypes.guess_extension(mime_type)
                if file_extension is None:
                    file_extension = ".wav"
                    wav_data = self._convert_to_wav(audio_data, mime_type)
                else:
                    wav_data = audio_data
                
                # Save temporary WAV file with unique timestamp to avoid conflicts
                tmp_wav = f"{base}_chunk_{i:03d}_{int(time.time())}.wav"
                temp_files.append(tmp_wav)
                self._save_binary_file(tmp_wav, wav_data)
                
                # Load and add to segments list
                try:
                    segment = AudioSegment.from_wav(tmp_wav)
                    audio_segments.append(segment)
                    logging.info(f"Added chunk {i + 1} to audio segments ({len(segment)} ms)")
                except Exception as e:
                    logging.error(f"Failed to load audio segment from {tmp_wav}: {e}")
                    raise

            # Concatenate all segments in order
            if audio_segments:
                logging.info(f"Combining {len(audio_segments)} audio segments...")
                combined = audio_segments[0]
                for segment in audio_segments[1:]:
                    combined += segment
                
                # Export final audio
                combined.export(output_path, format="mp3")
                logging.info(f"✓ Combined audio saved to {output_path} ({len(combined)} ms total)")
                
                # Cleanup temporary files
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        logging.warning(f"Could not remove temp file {temp_file}: {e}")
                
            else:
                raise RuntimeError("No audio segments were created")

        except Exception as e:
            logging.error(f"Gemini Flash TTS failed: {e}")
            # Cleanup temp files on error
            base, _ = os.path.splitext(output_path)
            for i in range(100):  # Clean up any leftover temp files
                temp_pattern = f"{base}_chunk_{i:03d}_*.wav"
                import glob
                for temp_file in glob.glob(temp_pattern):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            raise

    def _gtts_offline(self, text, output_path, chunk_size=4500):
        """
        Use gTTS offline to convert text to speech with improved chunking.
        """
        try:
            # Use smart chunking for better sentence boundaries
            chunks = self._smart_text_chunking(text, chunk_size)
            logging.info(f"Split into {len(chunks)} chunks for gTTS.")

            # Generate audio for each chunk
            if len(chunks) == 1:
                tts = gTTS(text=chunks[0], lang=self.language, slow=False)
                tts.save(output_path)
            else:
                audio_segments = []
                temp_files = []
                
                for i, chunk in enumerate(chunks):
                    chunk_path = f"{output_path.rsplit('.', 1)[0]}_gtts_{i:03d}.mp3"
                    temp_files.append(chunk_path)
                    
                    tts = gTTS(text=chunk, lang=self.language, slow=False)
                    tts.save(chunk_path)
                    audio_segments.append(AudioSegment.from_mp3(chunk_path))
                    logging.info(f"Generated chunk {i + 1}/{len(chunks)}")

                # Combine all audio segments
                combined = sum(audio_segments)
                combined.export(output_path, format="mp3")

                # Clean up temporary files
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

            logging.info(f"✓ Audio saved to {output_path} using gTTS offline.")
            
        except Exception as e:
            logging.error(f"gTTS offline TTS failed: {e}")
            raise

    def text_to_audio(self, text, output_path, chunk_size=5000, 
                     max_workers=5):
        """
        Convert text to speech using the selected TTS method.
        
        Parameters:
        - text: Input text to convert to speech
        - output_path: Path to save the combined audio file
        - chunk_size: Maximum byte size for each request (default: 5000)
        - style_instructions: Style instructions for speech generation (Google TTS only)
        - max_workers: Maximum parallel workers for Google TTS (default: 3)
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        if self.use_google:
            self._google_tts(
                text, output_path, 
                chunk_size=chunk_size, 
                max_workers=max_workers
            )
        else:
            self._gtts_offline(text, output_path, chunk_size=chunk_size)