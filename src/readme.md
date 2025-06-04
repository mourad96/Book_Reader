# Image Processor Module

This module provides functionality for processing PDFs and extracting text from images, with advanced footer detection and cropping features.

## Features
- Convert PDF pages to cropped images.
- Detect and crop footer regions dynamically using Hough Line Transform and adaptive thresholding.
- Extract text from images using Google Gemini Vision.
- Process a range of PDF pages and save the output in a specified directory.

## Installation
1. Ensure the correct installation of dependencies:
   - `google-generativeai`, `PyMuPDF` (`fitz`), `Pillow`, and `OpenCV`.


## Usage Instructions
1. **PDF to Cropped Images**:
   - Use the `convert_pdf_to_cropped_images` function to convert a PDF into cropped images.
   - Check the output directory for generated images.

2. **Manual Review**:
   - **Ensure that all cropped images do not include page numbers, bibliography, or any footer content** that might interfere with text extraction.
   - If any cropped images include footer content (e.g., page numbers or bibliographies), manually correct the cropping to ensure clean input for text extraction.

## Notes
- Failure to review and correct badly cropped images may lead to:
  - Retained page numbers or footer text in the cropped image.
  - Retained bibliography or other footer elements that may interfere with text extraction and subsequent processing.
- **Manually correcting such issues is critical for the optimal performance of this module.**


# TextToSpeech Module

The TextToSpeech module provides functionality to convert text into speech, with support for both Google Cloud Text-to-Speech (high-quality TTS with multiple languages and dialects) and gTTS (a simpler, offline solution).

## Features

- Supports Arabic
- Automatically handles large text inputs by chunking them into manageable pieces
- Combines multiple audio chunks into a single output file
- Allows switching between Google Cloud TTS and gTTS

## Requirements

### Dependencies

Install the required libraries:

```bash
pip install google-cloud-texttospeech pydub gtts python-dotenv
```

### Additional Setup for Google Cloud TTS

1. **Enable the API:**
   - Go to the Google Cloud Console
   - Enable the Text-to-Speech API

2. **Create and Download Service Account Key:**
   - Create a service account key and download the JSON file
   - Save it securely and set the path in a .env file (see below)

3. **Set Environment Variable:**
   Create a `.env` file in the project directory:
   ```bash
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your-service-account-file.json
   ```

## Usage

### Importing the Module

```python
from text_to_speech import TextToSpeech
```
```

## Key Methods

### TextToSpeech

Initialize the TextToSpeech class.
- `language` (str): Language code (e.g., 'ar' for Arabic)
- `use_google` (bool): Whether to use Google Cloud TTS (True) or gTTS (False)

### text_to_audio

Convert text into audio and save it to a file.

**Parameters:**
- `text` (str): The text to convert into speech
- `output_path` (str): The path to save the audio file
- `chunk_size` (int, optional): Maximum byte size for each chunk. Defaults to 4800 for Google Cloud TTS

**Usage:**
- Handles large inputs by splitting them into chunks
- Combines chunks into a single audio file

## How It Works

1. **Preprocessing:**
   - Normalizes text by removing extra spaces and unnecessary punctuation
   - Splits large text into smaller chunks based on Google Cloud's 5000-byte limit

2. **Audio Generation:**
   - Uses either Google Cloud TTS (for high-quality, cloud-based TTS) or gTTS (offline, simpler solution)

3. **Audio Combination:**
   - Combines multiple audio chunks into a single output file using pydub

4. **Cleanup:**
   - Removes temporary chunk files after combining them

## Error Handling

- Logs all errors for easier debugging
- Raises exceptions for missing environment variables or invalid configurations

## Notes

### Google Cloud TTS:
- Requires a valid service account and credentials file
- Supports multiple Arabic dialects

### gTTS:
- Simpler setup but limited to fewer voices and offline processing

## Contributing

Feel free to submit issues or pull requests to enhance the functionality.

