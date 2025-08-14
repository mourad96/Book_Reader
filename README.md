# Arabic Book Reader

A Python-based tool designed for processing and reading Arabic books in PDF format. This tool transforms your PDF books into accessible formats by extracting text using OCR, generating summaries, and creating audio versions for listening. Perfect for students, researchers, and book enthusiasts who need flexible ways to consume Arabic literature.

## Features

- PDF to image conversion
- OCR text extraction using Google Gemini Vision (supports Arabic)
- Text summarization via external API
- Text-to-speech conversion
- Configurable page range processing

## Prerequisites

- Python 3.8+
- API credentials for the summarization service
- Google Generative AI API key
- ffmpeg installed



### Installing ffmpeg
You may also need to install ffmpeg on your system:

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
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API credentials from [app.on-demand.io](https://app.on-demand.io):
```env
GEMINI_API_KEY=your_api_key_here     # used for OCR and summarization
EXTERNAL_USER_ID=your_external_user_id_here  # from app.on-demand.io
```

Note: You'll need to create an account on [app.on-demand.io](https://app.on-demand.io) to get your API credentials. These credentials are used for the text summarization feature.

## Usage

1. Place your PDF file in the project root directory.

2. Run the script:
```bash
python main.py
```

### Configuration Options

You can modify the following parameters in `main.py`:

```python
main(
    file_dir="./",                    # Directory containing the PDF
    output_text_file="output.txt",    # Output file for extracted text
    summary_file="summary.txt",       # Output file for summary
    audio_file="output.mp3",          # Output audio file
    api_key=os.getenv("GEMINI_API_KEY"),       # Gemini API key
    external_user_id=os.getenv("EXTERNAL_USER_ID"),  # On-demand external user ID
    start_page=1,                     # Starting page number
    end_page=None,                    # Ending page number (None for all pages)
    summarize=True,                   # Enable/disable summarization
    generate_audio=True,              # Enable/disable audio generation
    generate_img=True,                # Enable/disable image generation
    extract_txt=False                 # Enable/disable text extraction
)
```

## Project Structure

```
project_root/
├── requirements.txt
├── .env
├── main.py
└── src/
    ├── __init__.py
    ├── image_processor.py    # PDF to image and OCR functionality
    ├── text_processor.py     # Text summarization functionality
    └── text_to_speech.py     # Audio generation functionality
```

## Output Files

- `images/`: Directory containing extracted PDF pages as images
- `output.txt`: Extracted text from the PDF
- `summary.txt`: Summarized version of the extracted text
- `output.mp3`: Audio version of the text

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [gTTS](https://github.com/pndurette/gTTS)
- [Google Generative AI](https://ai.google)
