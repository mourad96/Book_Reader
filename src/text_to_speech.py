import logging
import re
from gtts import gTTS

class TextToSpeech:
    def __init__(self, language='ar'):
        self.language = language

    def _preprocess_text(self, text):
        """
        Preprocess text to improve speech continuity:
        1. Remove extra whitespaces
        2. Replace multiple newlines with a single space
        3. Remove unnecessary punctuation that might cause pauses
        """
        # Remove extra whitespaces and normalize Arabic text
        text = re.sub(r'\s+', ' ', text)
        
        # Remove unnecessary punctuation that might cause unnatural pauses
        text = text.replace('؛', ',')  # Replace Arabic semicolon with comma
        text = text.replace('»', '')   # Remove opening quote
        text = text.replace('«', '')   # Remove closing quote
        
        # Add a slight pause between paragraphs
        text = re.sub(r'\n\n', '. ', text)
        text = re.sub(r'\n', ' ', text)
        
        return text.strip()

    def text_to_audio(self, text, output_path, chunk_size=4500):
        """
        Convert text to speech with improved continuity:
        1. Preprocess the text
        2. Split into chunks to handle large texts
        3. Generate multiple audio files if necessary

        Parameters:
        - text: Input text to convert to speech
        - output_path: Path to save the audio file
        - chunk_size: Maximum characters per TTS request
        """
        try:
            # Preprocess the text
            processed_text = self._preprocess_text(text)
            
            # If text is longer than chunk_size, split it
            if len(processed_text) > chunk_size:
                # Split text into chunks without breaking words
                chunks = []
                while processed_text:
                    # Try to split at a natural break
                    if len(processed_text) <= chunk_size:
                        chunks.append(processed_text)
                        break
                    
                    # Find the last space before chunk_size
                    split_point = processed_text.rfind(' ', 0, chunk_size)
                    if split_point == -1:
                        split_point = chunk_size
                    
                    chunks.append(processed_text[:split_point])
                    processed_text = processed_text[split_point:].strip()
            else:
                chunks = [processed_text]
            
            # Generate audio for each chunk
            if len(chunks) == 1:
                # Single chunk, use standard method
                tts = gTTS(text=chunks[0], lang=self.language, slow=False)
                tts.save(output_path)
            else:
                # Multiple chunks, concatenate audio files
                from pydub import AudioSegment
                audio_segments = []
                
                for i, chunk in enumerate(chunks):
                    chunk_path = f"{output_path.rsplit('.', 1)[0]}_{i}.mp3"
                    tts = gTTS(text=chunk, lang=self.language, slow=False)
                    tts.save(chunk_path)
                    audio_segments.append(AudioSegment.from_mp3(chunk_path))
                
                # Combine segments with a very short silence between them
                combined = AudioSegment.empty()
                for segment in audio_segments:
                    combined += segment
                    # Add a very short silence (100 ms) between segments
                    combined += AudioSegment.silent(duration=100)
                
                # Export the final audio
                combined.export(output_path, format="mp3")
                
                # Optional: Clean up temporary chunk files
                import os
                for i in range(len(chunks)):
                    temp_path = f"{output_path.rsplit('.', 1)[0]}_{i}.mp3"
                    os.remove(temp_path)
            
            print(f"Audio saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to generate audio: {e}")