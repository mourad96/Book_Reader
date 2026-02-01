# Arabic Book Reader

A Python-based tool designed for processing and reading Arabic books in PDF format. This tool transforms your PDF books into accessible formats by extracting text using OCR, generating summaries, and creating audio versions for listening. Perfect for students, researchers, and book enthusiasts who need flexible ways to consume Arabic literature.

## Features

- **Interactive Header & Footer Cropping**: Visually crop headers and footers from PDF pages or images using a web interface.
- **PDF to Image Conversion**: High-quality conversion of PDF pages for processing.
- **OCR Text Extraction**: Utilizes Google Gemini Vision for accurate Arabic text extraction.
- **Text Summarization**: Generates concise summaries of the extracted text.
- **Text-to-Speech**: Converts the summarized text into natural-sounding audio.
- **Configurable Processing**: Options to control page ranges and specific processing steps.

## Prerequisites

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/download.html) installed and added to system PATH.
- API credentials for:
    - Google Generative AI (Gemini)
    - Summarization service (on-demand.io)

### Installing ffmpeg
You may need to install ffmpeg on your system:

#### Windows
```bash
winget install ffmpeg
```

#### Linux
```bash
sudo apt-get install ffmpeg
```

#### Mac
```bash
brew install ffmpeg
```

## Installation

1. Clone the repository:
```bash
git clone git@github.com:mourad96/Book_Reader.git
cd Book_Reader
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your credentials:
```env
GEMINI_API_KEY=your_api_key_here
EXTERNAL_USER_ID=your_external_user_id_here
```

## Usage Workflow

The typical workflow involves two main stages: Pre-processing (Cropping) and Data Extraction/Generation.

### Stage 1: Interactive Cropping (Fallback)
Use the interactive app to manually remove headers and footers if the automatic `image_processor.py` fails to produce the desired results.

1. Run the cropper app:
   ```bash
   streamlit run footer_cropper_app.py
   ```
2. **Upload** your PDF or images in the opened browser window.
3. **Adjust Lines**: Move the Green (Header) and Red (Footer) lines to exclude unwanted text.
4. **Batch Apply**: Use "Apply Top to ALL" or "Apply Bottom to ALL" for consistent layouts.
5. **Download**: Click "Process & Download All Images" to get a ZIP file.
6. **Extract**: Unzip the downloaded images into the `images/` directory in the project root.

### Stage 2: Text & Audio Generation
Once your images are ready (either from Stage 1 or from raw PDF conversion), run the main script.

```bash
python main.py
```

### Configuration Options in `main.py`
You can modify the `main()` function parameters in `main.py` to control the pipeline:

```python
main(
    file_dir="./",                    # Directory containing the original PDF (if starting from PDF)
    output_text_file="output.txt",    # Output file for extracted text
    summary_file="summary.txt",       # Output file for summary
    audio_file="output.mp3",          # Output audio file
    api_key=os.getenv("GEMINI_API_KEY"),
    external_user_id=os.getenv("EXTERNAL_USER_ID"),
    
    start_page=1,                     # Start page (relevant if generating images from PDF)
    end_page=None,                    # End page
    
    # Flags to control steps:
    generate_img=False,               # Set to True to generate images from PDF. Set to False if using cropped images in images/ dir.
    extract_txt=True,                 # Set to True to extract text from images in the images/ dir.
    summarize=True,                   # Enable/disable summarization.
    generate_audio=True               # Enable/disable audio generation.
)
```

**Common Scenarios:**

*   **Scenario A: Full Pipeline (PDF -> Audio)**
    *   Set `generate_img=True`, `extract_txt=True`, `summarize=True`, `generate_audio=True`.
    *   *Note: This skips the manual cropping step.*

*   **Scenario B: Using Cropped Images (Cropper -> Audio)**
    *   Place cropped images in `images/`.
    *   Set `generate_img=False` (prevents overwriting your cropped images).
    *   Set `extract_txt=True`, `summarize=True`, `generate_audio=True`.

## Project Structure

```
project_root/
├── requirements.txt
├── .env
├── main.py                  # Main orchestration script
├── footer_cropper_app.py    # Interactive Streamlit app for cropping
└── src/
    ├── __init__.py
    ├── image_processor.py   # PDF processing & OCR
    ├── text_processor.py    # Summarization logic
    └── text_to_speech.py    # Audio generation
```

## Output Files

- `images/`: Directory for processing images.
- `output.txt`: Raw text extracted from images.
- `summary.txt`: Generated summary of the text.
- `output.mp3`: Final audio file.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [gTTS](https://github.com/pndurette/gTTS)
- [Google Generative AI](https://ai.google)
