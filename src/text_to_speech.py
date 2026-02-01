# from google.cloud import texttospeech
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
import re as regex_module


class RateLimiter:
    """
    Rate limiter to enforce API request limits.
    Default: 10 requests per minute for Gemini TTS free tier.
    
    Configuration:
    - Max 10 requests per 60 seconds (10/min)
    - No minimum delay between requests - send as fast as possible
    - Only wait when the 10 RPM limit is actually reached
    """
    def __init__(self, max_requests=10, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait only if we've hit the rate limit (10 requests in the last 60 seconds)."""
        with self.lock:
            now = time.time()
            
            # Remove requests older than the time window
            self.requests = [t for t in self.requests if now - t < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # Calculate wait time until oldest request expires from the window
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request) + 1  # +1 buffer
                if wait_time > 0:
                    logging.info(f"⏳ Rate limit reached ({len(self.requests)}/{self.max_requests} in last {self.time_window}s). "
                               f"Waiting {wait_time:.1f}s for oldest request to expire...")
                    time.sleep(wait_time)
                    # Clear old requests after waiting
                    now = time.time()
                    self.requests = [t for t in self.requests if now - t < self.time_window]
            
            # Record this request
            self.requests.append(now)
            logging.info(f"Request window: {len(self.requests)}/{self.max_requests} requests in last {self.time_window}s")


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
        Improved text chunking that respects sentence boundaries while ensuring:
        1. Chunks are packed close to target size (minimum 80% threshold)
        2. Hard maximum of 4000 bytes is never exceeded
        3. Sentences are split by words if they exceed the limit
        
        Parameters:
        - text: Input text to chunk
        - max_byte_size: Target byte size per chunk (soft limit)
        
        Returns:
        - List of text chunks
        """
        # Hard maximum that is never exceeded (for API/quality limits)
        HARD_MAX_BYTES = 4000
        
        # Use the smaller of user's max or hard max
        effective_max = min(max_byte_size, HARD_MAX_BYTES)
        
        # Minimum threshold before starting a new chunk (80% of target)
        min_threshold = int(effective_max * 0.8)
        
        # Preprocess the text first
        processed_text = text
        
        # Split into sentences first to respect natural boundaries
        # Using a pattern that keeps the sentence ending with the sentence
        sentence_pattern = re.compile(r'(?<=[.!?؟])\s+')
        sentences = sentence_pattern.split(processed_text)
        
        chunks = []
        current_chunk = ""
        current_chunk_bytes = 0
        
        def get_byte_size(s):
            return len(s.encode('utf-8'))
        
        def split_by_words(text_to_split, max_bytes):
            """Split text by words, ensuring no chunk exceeds max_bytes."""
            words = text_to_split.split()
            result_chunks = []
            temp_chunk = ""
            
            for word in words:
                test_chunk = temp_chunk + " " + word if temp_chunk else word
                
                if get_byte_size(test_chunk) <= max_bytes:
                    temp_chunk = test_chunk
                else:
                    if temp_chunk:
                        result_chunks.append(temp_chunk.strip())
                    # If single word exceeds max, we have to include it anyway
                    if get_byte_size(word) > max_bytes:
                        result_chunks.append(word)
                        temp_chunk = ""
                    else:
                        temp_chunk = word
            
            if temp_chunk:
                result_chunks.append(temp_chunk.strip())
            
            return result_chunks
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_bytes = get_byte_size(sentence)
            
            # If sentence alone exceeds the hard max, split it by words
            if sentence_bytes > effective_max:
                # First, save current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_chunk_bytes = 0
                
                # Split the long sentence and add each part
                split_parts = split_by_words(sentence, effective_max)
                for part in split_parts:
                    chunks.append(part)
                continue
            
            # Check if adding this sentence would exceed the limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            test_chunk_bytes = get_byte_size(test_chunk)
            
            if test_chunk_bytes <= effective_max:
                # Sentence fits, add it to current chunk
                current_chunk = test_chunk
                current_chunk_bytes = test_chunk_bytes
            else:
                # Sentence doesn't fit - decide whether to start new chunk
                # Only start new chunk if current is at least at minimum threshold
                if current_chunk_bytes >= min_threshold:
                    # Current chunk is full enough, save it and start new
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_chunk_bytes = sentence_bytes
                else:
                    # Current chunk is too small, try to fit more by splitting
                    # Calculate how much space we have left
                    remaining_space = effective_max - current_chunk_bytes - 1  # -1 for space
                    
                    if remaining_space > 100:  # Only split if meaningful space available
                        # Try to fit part of the sentence
                        words = sentence.split()
                        fitting_part = ""
                        remaining_words = []
                        
                        for i, word in enumerate(words):
                            test_part = fitting_part + " " + word if fitting_part else word
                            if get_byte_size(current_chunk + " " + test_part) <= effective_max:
                                fitting_part = test_part
                            else:
                                remaining_words = words[i:]
                                break
                        
                        if fitting_part:
                            current_chunk = current_chunk + " " + fitting_part if current_chunk else fitting_part
                            chunks.append(current_chunk.strip())
                            current_chunk = " ".join(remaining_words) if remaining_words else ""
                            current_chunk_bytes = get_byte_size(current_chunk)
                        else:
                            # Can't fit anything, just save current and start new
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                            current_chunk_bytes = sentence_bytes
                    else:
                        # Not enough space, save current chunk and start new
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                        current_chunk_bytes = sentence_bytes
        
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

    def _parse_retry_delay(self, error_message):
        """
        Parse retry delay from API error message.
        
        Returns:
        - float: Retry delay in seconds, or default of 30 if not found
        """
        try:
            # Look for patterns like "Please retry in 29.944153256s."
            match = regex_module.search(r'retry in ([\d.]+)s', str(error_message))
            if match:
                return float(match.group(1)) + 2  # Add 2 second buffer
            
            # Look for retryDelay in the error
            match = regex_module.search(r"'retryDelay':\s*'(\d+)s'", str(error_message))
            if match:
                return float(match.group(1)) + 2
        except:
            pass
        return 32  # Default: 32 seconds

    def _process_single_chunk(self, chunk_index, chunk_text, model, voice_name, client, 
                            style_instructions=None, voice_config_overrides=None,
                            rate_limiter=None, max_retries_429=10):
        """
        Process a single chunk with consistent voice parameters and rate limit handling.
        
        Parameters:
        - chunk_index: Index of the chunk being processed
        - chunk_text: Text content to process
        - model: Model name for TTS
        - voice_name: Voice name to use
        - client: API client
        - style_instructions: Style instructions to apply to this chunk
        - voice_config_overrides: Additional voice configuration parameters
        - rate_limiter: RateLimiter instance for enforcing rate limits
        - max_retries_429: Maximum retries for 429 errors
        
        Returns:
        - tuple: (chunk_index, success, audio_data, mime_type, error_message)
        """
        # Apply rate limiting before making request
        if rate_limiter:
            rate_limiter.wait_if_needed()
        
        for retry_429 in range(max_retries_429):
            try:
                logging.info(f"Processing chunk {chunk_index + 1}: '{chunk_text[:50]}...'")
                
                # Apply style instructions to every chunk for consistency
                if style_instructions:
                    # Create a consistent prompt structure
                    formatted_text = f"{style_instructions}\n\nText to read: {chunk_text}"
                else:
                    formatted_text = chunk_text
                
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=formatted_text)],
                    ),
                ]
                
                # Build voice config with consistency parameters
                voice_config = types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
                
                # Apply any voice config overrides for consistency
                speech_config = types.SpeechConfig(voice_config=voice_config)
                
                # Add voice consistency parameters if available
                if voice_config_overrides:
                    # Apply additional parameters for voice consistency
                    # These might include pitch, speed, etc. depending on API capabilities
                    pass
                
                generate_content_config = types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=speech_config
                )

                audio_data = b""
                mime_type = None
                
                # Use non-streaming API call for consistent audio output
                # Streaming can produce packets with varying sample rates/timing,
                # causing jittery or accelerated audio when concatenated
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                # Extract audio from non-streaming response
                if (response.candidates and 
                    response.candidates[0].content and 
                    response.candidates[0].content.parts):
                    for part in response.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.data:
                            audio_data += part.inline_data.data
                            if mime_type is None:
                                mime_type = part.inline_data.mime_type

                if not audio_data:
                    return chunk_index, False, None, None, "No audio data received"
                
                logging.info(f"Successfully processed chunk {chunk_index + 1} ({len(audio_data)} bytes)")
                return chunk_index, True, audio_data, mime_type, None
                        
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error (429)
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                    # Exponential backoff: base_delay * (2 ^ retry_attempt)
                    base_delay = self._parse_retry_delay(error_str)
                    exponential_delay = base_delay * (2 ** retry_429)  # 32, 64, 128, 256...
                    # Cap at 5 minutes max
                    exponential_delay = min(exponential_delay, 300)
                    
                    logging.warning(f"⚠️ Rate limit hit for chunk {chunk_index + 1}. "
                                  f"Exponential backoff: waiting {exponential_delay:.1f}s "
                                  f"(retry {retry_429 + 1}/{max_retries_429})...")
                    time.sleep(exponential_delay)
                    
                    # Reset rate limiter tracking after long wait
                    if rate_limiter:
                        with rate_limiter.lock:
                            rate_limiter.requests = []  # Clear request history after long wait
                    continue
                else:
                    # Non-rate-limit error, return immediately
                    error_msg = f"Error processing chunk {chunk_index}: {error_str}"
                    logging.error(error_msg)
                    return chunk_index, False, None, None, error_msg
        
        # All 429 retries exhausted
        error_msg = f"Chunk {chunk_index} failed after {max_retries_429} rate limit retries"
        logging.error(error_msg)
        return chunk_index, False, None, None, error_msg

    def _google_tts(self, text, output_path, chunk_size=1500,
                model="gemini-2.5-flash-preview-tts", voice_name="Algieba", 
                style_instructions="You are a professional narrator. Deliver the text with a natural, engaging storytelling tone and clear articulation, using subtle intonation changes at commas and sentence endings.", 
                max_workers=3, max_retries=3, voice_config_overrides=None,
                keep_chunks=False, chunks_dir=None):
        """
        Use Google Gemini Flash TTS with improved error handling, chunk management, and voice consistency.
        """
        try:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set in .env")
            
            client = genai.Client(api_key=api_key)

            # Smart chunking that respects sentence boundaries
            chunks = self._smart_text_chunking(text, chunk_size)
            
            logging.info(f"Split text into {len(chunks)} chunks for Gemini TTS.")
            for i, chunk in enumerate(chunks):
                logging.info(f"Chunk {i + 1} length: {len(chunk)} chars, {len(chunk.encode('utf-8'))} bytes")
            
            # Validate chunk count against daily API limit (100 requests/day)
            MAX_DAILY_REQUESTS = 100
            if len(chunks) > MAX_DAILY_REQUESTS:
                # Calculate the minimum byte size per chunk to stay under 100 chunks
                total_text_bytes = len(text.encode('utf-8'))
                suggested_chunk_size = (total_text_bytes // MAX_DAILY_REQUESTS) + 1
                
                raise ValueError(
                    f"\n{'='*60}\n"
                    f"❌ TOO MANY CHUNKS: {len(chunks)} chunks exceeds the daily API limit of {MAX_DAILY_REQUESTS} requests!\n"
                    f"{'='*60}\n"
                    f"\n📊 Text Statistics:\n"
                    f"   - Total text size: {total_text_bytes:,} bytes\n"
                    f"   - Current chunk size: {chunk_size:,} bytes\n"
                    f"   - Resulting chunks: {len(chunks)}\n"
                    f"\n💡 SUGGESTION:\n"
                    f"   To stay under {MAX_DAILY_REQUESTS} chunks, set chunk_size to at least {suggested_chunk_size:,} bytes.\n"
                    f"   \n"
                    f"   Example: text_to_audio(text, output, chunk_size={suggested_chunk_size})\n"
                    f"\n⚠️  Alternatively, you can split your text into smaller parts and process\n"
                    f"   them across multiple days.\n"
                    f"{'='*60}"
                )
            
            # Warn if chunk size exceeds recommended limit for audio quality
            RECOMMENDED_MAX_CHUNK_SIZE = 4000
            if chunk_size > RECOMMENDED_MAX_CHUNK_SIZE:
                logging.warning(
                    f"\n{'⚠️ '*30}\n"
                    f"⚠️  AUDIO QUALITY WARNING\n"
                    f"{'⚠️ '*30}\n"
                    f"\n📢 Your chunk_size ({chunk_size:,} bytes) exceeds the recommended maximum of {RECOMMENDED_MAX_CHUNK_SIZE:,} bytes.\n"
                    f"\n🔊 Impact on Audio Quality:\n"
                    f"   - Larger chunks may result in deteriorated audio quality\n"
                    f"   - The TTS model performs better with smaller, focused text segments\n"
                    f"   - Voice consistency and naturalness may be affected\n"
                    f"\n💡 Recommendation:\n"
                    f"   Consider using chunk_size ≤ {RECOMMENDED_MAX_CHUNK_SIZE:,} bytes for optimal audio quality.\n"
                    f"   Current setting will proceed, but quality may vary.\n"
                    f"{'⚠️ '*30}\n"
                )

            # Dictionary to store results with guaranteed ordering
            chunk_results = {}
            base, _ = os.path.splitext(output_path)
            
            # Setup separate chunks directory if requested
            if keep_chunks and chunks_dir:
                os.makedirs(chunks_dir, exist_ok=True)
            
            failed_chunks = []
            
            # Create a consistent voice configuration that will be used for all chunks
            consistent_voice_config = voice_config_overrides or {}
            
            # Rate limiter: 10 requests per minute - sends as fast as possible, only waits when limit is hit
            rate_limiter = RateLimiter(max_requests=10, time_window=60)
            logging.info(f"Rate limiter configured: {rate_limiter.max_requests} requests per {rate_limiter.time_window}s (no delay until limit reached)")
            
            # Process chunks in parallel (up to max_workers threads)
            # Rate limiter will block threads when 10 RPM limit is reached
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit chunks for processing with consistent parameters
                future_to_index = {}
                for i, chunk in enumerate(chunks):
                    future = executor.submit(
                        self._process_single_chunk, 
                        i, chunk, model, voice_name, client,
                        style_instructions=style_instructions,  # Apply to ALL chunks
                        voice_config_overrides=consistent_voice_config,
                        rate_limiter=rate_limiter,
                        max_retries_429=5
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

            # Retry failed chunks sequentially with same voice parameters
            for retry_attempt in range(max_retries):
                if not failed_chunks:
                    break
                    
                logging.info(f"Retry attempt {retry_attempt + 1}/{max_retries} for {len(failed_chunks)} failed chunks")
                retry_failed = []
                
                for chunk_index in failed_chunks:
                    # Rate limiter handles waiting
                    chunk_index, success, audio_data, mime_type, error_msg = self._process_single_chunk(
                        chunk_index, chunks[chunk_index], model, voice_name, client,
                        style_instructions=style_instructions,  # Consistent style for retries
                        voice_config_overrides=consistent_voice_config,
                        rate_limiter=rate_limiter,
                        max_retries_429=5
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

            # Process results in correct order with audio normalization for consistency
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
                
                # Explicitly save to chunks_dir if requested
                if keep_chunks and chunks_dir:
                    chunk_filename = f"chunk_{i:03d}{file_extension}"
                    chunk_path = os.path.join(chunks_dir, chunk_filename)
                    self._save_binary_file(chunk_path, wav_data)
                    logging.info(f"Saved chunk {i + 1} to {chunk_path}")

                    # Also save the text content
                    text_filename = f"chunk_{i:03d}.txt"
                    text_path = os.path.join(chunks_dir, text_filename)
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(chunks[i])
                    logging.info(f"Saved text chunk {i + 1} to {text_path}")
                
                # Save temporary WAV file for combination (managed separately from kept chunks)
                tmp_wav = f"{base}_chunk_{i:03d}_{int(time.time())}.wav"
                temp_files.append(tmp_wav)
                self._save_binary_file(tmp_wav, wav_data)
                
                # Load and normalize audio segment for consistency
                try:
                    segment = AudioSegment.from_wav(tmp_wav)
                    
                    # Apply audio normalization for voice consistency
                    # Normalize volume levels between chunks
                    normalized_segment = segment.normalize()
                    
                    # Optional: Apply consistent audio processing
                    # You can add more audio processing here for consistency
                    
                    audio_segments.append(normalized_segment)
                    logging.info(f"Added normalized chunk {i + 1} to audio segments ({len(normalized_segment)} ms)")
                except Exception as e:
                    logging.error(f"Failed to load audio segment from {tmp_wav}: {e}")
                    raise

            # Concatenate all segments in order with smooth transitions
            if audio_segments:
                logging.info(f"Combining {len(audio_segments)} audio segments...")
                combined = audio_segments[0]
                
                for segment in audio_segments[1:]:
                    # Add very small crossfade to smooth transitions between chunks
                    combined = combined.append(segment, crossfade=50)  # 50ms crossfade
                
                # Final normalization for overall consistency
                combined = combined.normalize()
                
                # Export final audio
                combined.export(output_path, format="mp3")
                logging.info(f"✓ Combined audio saved to {output_path} ({len(combined)} ms total)")
                
                # Cleanup temporary files (only the ones created for combination)
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

    def _gtts_offline(self, text, output_path, chunk_size=4500, keep_chunks=False, chunks_dir=None):
        """
        Use gTTS offline to convert text to speech with improved chunking and consistency.
        """
        try:
            # Use smart chunking for better sentence boundaries
            chunks = self._smart_text_chunking(text, chunk_size)
            logging.info(f"Split into {len(chunks)} chunks for gTTS.")
            
            if keep_chunks and chunks_dir:
                os.makedirs(chunks_dir, exist_ok=True)

            # Generate audio for each chunk with consistent parameters
            if len(chunks) == 1:
                tts = gTTS(text=chunks[0], lang=self.language, slow=False)
                tts.save(output_path)
                
                if keep_chunks and chunks_dir:
                    chunk_path = os.path.join(chunks_dir, "chunk_000.mp3")
                    tts.save(chunk_path)
                    
            else:
                audio_segments = []
                temp_files = []
                
                for i, chunk in enumerate(chunks):
                    # Temp file for combination
                    chunk_path = f"{output_path.rsplit('.', 1)[0]}_gtts_{i:03d}.mp3"
                    temp_files.append(chunk_path)
                    
                    # Use consistent parameters for all chunks
                    tts = gTTS(text=chunk, lang=self.language, slow=False)
                    tts.save(chunk_path)
                    
                    # Save separate chunk if requested
                    if keep_chunks and chunks_dir:
                        kept_chunk_path = os.path.join(chunks_dir, f"chunk_{i:03d}.mp3")
                        tts.save(kept_chunk_path)
                    
                    # Load and normalize segment
                    segment = AudioSegment.from_mp3(chunk_path)
                    normalized_segment = segment.normalize()
                    audio_segments.append(normalized_segment)
                    logging.info(f"Generated chunk {i + 1}/{len(chunks)}")

                # Combine all audio segments with smooth transitions
                combined = audio_segments[0]
                for segment in audio_segments[1:]:
                    combined = combined.append(segment, crossfade=50)  # Small crossfade
                
                # Final normalization
                combined = combined.normalize()
                combined.export(output_path, format="mp3")

                # Clean up temporary files
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

            logging.info(f"✓ Audio saved to {output_path} using gTTS offline.")
            
        except Exception as e:
            logging.error(f"gTTS offline TTS failed: {e}")
            raise

    def text_to_audio(self, text, output_path, chunk_size=2800, 
                     max_workers=9, style_instructions=None, voice_config_overrides=None,
                     keep_chunks=True, chunks_dir=None):
        """
        Convert text to speech using the selected TTS method with improved voice consistency.
        
        Parameters:
        - text: Input text to convert to speech
        - output_path: Path to save the combined audio file
        - chunk_size: Maximum byte size for each request (default: 5000)
        - max_workers: Maximum parallel workers for Google TTS (default: 3)
        - style_instructions: Style instructions applied to ALL chunks for consistency
        - voice_config_overrides: Additional voice configuration for consistency
        - keep_chunks: If True, saves individual chunks to chunks_dir
        - chunks_dir: Directory to save individual chunks if keep_chunks is True. Defaults to 'audio' in project root.
        """
        if keep_chunks and chunks_dir is None:
            # Default to 'audio' folder in the project root (parent of src)
            chunks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        if self.use_google:
            self._google_tts(
                text, output_path, 
                chunk_size=chunk_size, 
                max_workers=max_workers,
                style_instructions=style_instructions,
                voice_config_overrides=voice_config_overrides,
                keep_chunks=keep_chunks,
                chunks_dir=chunks_dir
            )
        else:
            self._gtts_offline(
                text, output_path, 
                chunk_size=chunk_size,
                keep_chunks=keep_chunks,
                chunks_dir=chunks_dir
            )