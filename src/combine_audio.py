import os
import argparse
import logging
import subprocess
import tempfile
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def combine_audio_ffmpeg(input_dir, output_file, format='mp3'):
    """
    Combines all audio chunks using ffmpeg concat demuxer (more reliable).
    
    Parameters:
    - input_dir: Directory containing audio chunks.
    - output_file: Path to the output audio file.
    - format: Output format (default: mp3).
    """
    try:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        # Get list of audio files, sorted by name
        supported_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
        files = sorted([
            f for f in os.listdir(input_dir) 
            if any(f.lower().endswith(ext) for ext in supported_extensions)
        ])
        
        if not files:
            logging.warning(f"No audio files found in {input_dir}")
            return
            
        logging.info(f"Found {len(files)} chunks to combine using ffmpeg.")
        
        # Create a concat list file for ffmpeg
        concat_list_path = os.path.join(input_dir, '_concat_list.txt')
        
        with open(concat_list_path, 'w', encoding='utf-8') as f:
            for file in files:
                # Use absolute path with forward slashes for ffmpeg compatibility
                abs_path = os.path.abspath(os.path.join(input_dir, file)).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
        
        logging.info(f"Created concat list with {len(files)} files")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Build ffmpeg command
        # -f concat: use concat demuxer
        # -safe 0: allow absolute paths
        # -i: input concat list
        # -c:a libmp3lame: encode to mp3
        # -q:a 2: high quality VBR
        if format == 'mp3':
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_path,
                '-c:a', 'libmp3lame',
                '-q:a', '2',
                output_file
            ]
        elif format == 'wav':
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_path,
                '-c:a', 'pcm_s16le',
                '-ar', '24000',
                output_file
            ]
        else:
            # Generic copy for other formats
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_path,
                '-c:a', 'copy',
                output_file
            ]
        
        logging.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        # Clean up concat list
        try:
            os.remove(concat_list_path)
        except:
            pass
        
        if result.returncode != 0:
            logging.error(f"ffmpeg error: {result.stderr}")
            raise RuntimeError(f"ffmpeg failed with code {result.returncode}")
        
        # Get output file info
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            logging.info(f"Successfully combined audio saved to {output_file} ({size_mb:.2f} MB)")
        else:
            raise RuntimeError("Output file was not created")
            
    except Exception as e:
        logging.error(f"Failed to combine audio with ffmpeg: {e}")
        raise


def combine_audio_pydub(input_dir, output_file, format='mp3', target_sample_rate=24000):
    """
    Combines all audio chunks using pydub (original method).
    
    Parameters:
    - input_dir: Directory containing audio chunks.
    - output_file: Path to the output audio file.
    - format: Format of the audio files (default: mp3).
    - target_sample_rate: Target sample rate for consistency (default: 24000 Hz).
    """
    try:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        # Get list of files, sorted by name to ensure correct order
        supported_extensions = [f".{format}", ".wav"]
        files = sorted([
            f for f in os.listdir(input_dir) 
            if any(f.lower().endswith(ext) for ext in supported_extensions)
        ])
        
        if not files:
            logging.warning(f"No audio files found in {input_dir}")
            return
            
        logging.info(f"Found {len(files)} chunks to combine using pydub.")
        
        audio_segments = []
        
        for i, file in enumerate(files):
            file_path = os.path.join(input_dir, file)
            logging.info(f"Processing {file}...")
            
            try:
                # Detect format from file extension
                file_ext = os.path.splitext(file)[1].lower().lstrip('.')
                segment = AudioSegment.from_file(file_path, format=file_ext)
                
                # Normalize sample rate to prevent timing issues
                if segment.frame_rate != target_sample_rate:
                    logging.info(f"  Resampling from {segment.frame_rate}Hz to {target_sample_rate}Hz")
                    segment = segment.set_frame_rate(target_sample_rate)
                
                # Ensure consistent audio format (mono, 16-bit)
                if segment.channels != 1:
                    logging.info(f"  Converting from {segment.channels} channels to mono")
                    segment = segment.set_channels(1)
                if segment.sample_width != 2:  # 16-bit = 2 bytes
                    logging.info(f"  Converting from {segment.sample_width * 8}-bit to 16-bit")
                    segment = segment.set_sample_width(2)
                
                # Normalize volume for this segment
                segment = segment.normalize()
                
                audio_segments.append(segment)
                logging.info(f"  Added chunk {i + 1}: {len(segment)}ms, {segment.frame_rate}Hz")
                    
            except Exception as e:
                logging.error(f"Error processing {file}: {e}")
                # Continue with other files but log the error
        
        if not audio_segments:
            logging.error("No audio segments could be loaded")
            return
        
        # Combine all segments with crossfade
        combined = audio_segments[0]
        for segment in audio_segments[1:]:
            combined = combined.append(segment, crossfade=50)  # 50ms crossfade
        
        # Final normalization for overall consistency
        combined = combined.normalize()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        combined.export(output_file, format=format)
        logging.info(f"Successfully combined audio saved to {output_file} ({len(combined)}ms total)")
        
    except Exception as e:
        logging.error(f"Failed to combine audio with pydub: {e}")
        raise


def combine_audio_chunks(input_dir, output_file, format='mp3', use_ffmpeg=True, target_sample_rate=24000):
    """
    Combines all audio chunks in the specified directory into a single audio file.
    
    Parameters:
    - input_dir: Directory containing audio chunks.
    - output_file: Path to the output audio file.
    - format: Format of the audio files (default: mp3).
    - use_ffmpeg: If True, use ffmpeg concat (recommended). If False, use pydub.
    - target_sample_rate: Target sample rate for pydub method (default: 24000 Hz).
    """
    if use_ffmpeg:
        combine_audio_ffmpeg(input_dir, output_file, format)
    else:
        combine_audio_pydub(input_dir, output_file, format, target_sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine audio chunks from a directory.")
    parser.add_argument("input_dir", help="Directory containing audio chunks")
    parser.add_argument("output_file", help="Path to the output combined audio file")
    parser.add_argument("--format", default="mp3", help="Audio format (default: mp3)")
    parser.add_argument("--use-pydub", action="store_true", 
                        help="Use pydub instead of ffmpeg (ffmpeg is default and recommended)")
    
    args = parser.parse_args()
    
    combine_audio_chunks(
        args.input_dir, 
        args.output_file, 
        args.format,
        use_ffmpeg=not args.use_pydub
    )
