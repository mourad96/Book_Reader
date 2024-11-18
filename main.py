import pytesseract
from PIL import Image
import os
from gtts import gTTS
from pathlib import Path
import logging
import requests
import fitz 
from dotenv import load_dotenv

load_dotenv()

class ImageProcessor:
    def __init__(self, tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def convert_pdf_to_images(self, pdf_path, output_dir, start_page=1, end_page=None, dpi=300):
        """
        Convert a specific range of pages from a PDF to images.

        Parameters:
        - pdf_path: Path to the PDF file.
        - output_dir: Directory to save the images.
        - start_page: First page to process (1-indexed).
        - end_page: Last page to process (inclusive, 1-indexed). If None, process until the last page.
        - dpi: DPI for the rendered images.
        """
        try:
            with fitz.open(pdf_path) as pdf:
                total_pages = len(pdf)
                end_page = end_page or total_pages  # If no end_page, process until the last page

                # Validate page range
                if start_page < 1 or end_page > total_pages or start_page > end_page:
                    raise ValueError("Invalid page range specified.")

                for page_number in range(start_page - 1, end_page):
                    page = pdf[page_number]
                    pix = page.get_pixmap(dpi=dpi)
                    image_path = os.path.join(output_dir, f"{page_number + 1}.png")  # Save as 1.png, 2.png, etc.
                    pix.save(image_path)
                    logging.info(f"Saved page {page_number + 1} of {pdf_path} as {image_path}")
        except Exception as e:
            logging.error(f"Failed to convert PDF to images: {e}")

    @staticmethod
    def get_sorted_image_files(directory):
        """Fetch and sort image files by numeric prefix."""
        valid_extensions = (".png", ".jpg")
        files = [filename for filename in os.listdir(directory) if filename.endswith(valid_extensions)]

        def extract_numeric_prefix(filename):
            """Extract numeric prefix from filename."""
            try:
                return int(filename.split('.')[0])
            except ValueError:
                return float('inf')  # Non-numeric files go to the end

        sorted_files = sorted(files, key=extract_numeric_prefix)
        return [os.path.join(directory, filename) for filename in sorted_files]

    def extract_text_from_images(self, image_files, lang='ara'):
        """Extract text from a list of sorted image files."""
        text = ""
        for image_path in image_files:
            try:
                text += pytesseract.image_to_string(Image.open(image_path), lang=lang, config="--psm 6")
            except Exception as e:
                logging.error(f"Failed to process {image_path}: {e}")
        return text

    def extract_text_from_pdf(self, pdf_path, output_dir, start_page=1, end_page=None, lang='ara'):
        """Convert a range of pages from a PDF to images and extract text."""
        self.convert_pdf_to_images(pdf_path, output_dir, start_page, end_page)
        sorted_images = self.get_sorted_image_files(output_dir)
        return self.extract_text_from_images(sorted_images, lang)

class TextProcessor:
    def __init__(self, api_key, external_user_id):
        self.api_key = api_key
        self.external_user_id = external_user_id

    def create_session(self):
        """Create a chat session."""
        create_session_url = 'https://api.on-demand.io/chat/v1/sessions'
        create_session_headers = {'apikey': self.api_key}
        create_session_body = {
            "pluginIds": [],
            "externalUserId": self.external_user_id
        }

        response = requests.post(create_session_url, headers=create_session_headers, json=create_session_body)
        if response.status_code != 201:
            raise Exception(f"Failed to create session: {response.text}")
        return response.json()['data']['id']

    def submit_query(self, session_id, text):
        """Submit a query to the session."""
        submit_query_url = f'https://api.on-demand.io/chat/v1/sessions/{session_id}/query'
        submit_query_headers = {'apikey': self.api_key}
        submit_query_body = {
            "endpointId": "predefined-openai-gpt4o",
            "query": f"Please summarize the following text:\n{text}",
            "pluginIds": ["plugin-1712327325", "plugin-1713962163"],
            "responseMode": "sync"
        }

        response = requests.post(submit_query_url, headers=submit_query_headers, json=submit_query_body)
        if response.status_code != 200:
            raise Exception(f"Failed to submit query: {response.text}")
        return response.json()['data']['answer']

    def summarize_text(self, text):
        """Summarize text using the external API."""
        try:
            logging.info("Creating chat session for summarization...")
            session_id = self.create_session()
            logging.info(f"Chat session created with ID: {session_id}")
            
            logging.info("Submitting text to the external API for summarization...")
            summary = self.submit_query(session_id, text)
            return summary
        except Exception as e:
            logging.error(f"Failed to summarize text: {e}")
            return text

class TextToSpeech:
    def __init__(self, language='ar'):
        self.language = language

    def text_to_audio(self, text, output_path):
        """Convert text to speech and save as an audio file."""
        try:
            tts = gTTS(text=text, lang=self.language, slow=False)
            tts.save(output_path)
            print(f"Audio saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to generate audio: {e}")

def main(file_dir, output_text_file, summary_file, audio_file, tesseract_cmd, api_key, external_user_id, 
         start_page=None, end_page=None, summarize=False, generate_audio=False):
    logging.basicConfig(level=logging.INFO)
    processor = ImageProcessor(tesseract_cmd)
    text_processor = TextProcessor(api_key, external_user_id)
    synthesizer = TextToSpeech()

    # Ensure file_dir contains only one PDF to process
    pdf_files = [f for f in os.listdir(file_dir) if f.endswith('.pdf')]
    if len(pdf_files) != 1:
        raise ValueError("Please ensure the directory contains exactly one PDF to process.")

    pdf_path = os.path.join(file_dir, pdf_files[0])
    output_dir = "./images"
    os.makedirs(output_dir, exist_ok=True)

    # Extract text from the specified range of the PDF
    logging.info(f"Processing PDF: {pdf_path}, Pages: {start_page}-{end_page}")
    extracted_text = processor.extract_text_from_pdf(
        pdf_path=pdf_path, 
        output_dir=output_dir, 
        start_page=start_page, 
        end_page=end_page
    )

    # Save the extracted text
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    logging.info(f"Extracted text saved to {output_text_file}")

    # Summarize text if the option is enabled
    if summarize:
        logging.info("Summarizing the extracted text...")
        summarized_text = text_processor.summarize_text(extracted_text)

        # Save the summarized text
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summarized_text)
        logging.info(f"Summarized text saved to {summary_file}")
    else:
        summarized_text = extracted_text

    # Convert text to audio if the option is enabled
    if generate_audio:
        synthesizer.text_to_audio(extracted_text, audio_file)
        logging.info(f"Audio file saved to {audio_file}")

if __name__ == "__main__":
    main(
        file_dir="./",
        output_text_file="output.txt",
        summary_file="summary.txt",
        audio_file="output.mp3",
        tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        api_key=os.getenv("API_KEY"),  # Replace with your actual API key
        external_user_id=os.getenv("EXTERNAL_USER_ID"),  # Replace with your actual external user ID
        start_page=320,  # Specify starting page
        end_page=322,  # Specify ending page
        summarize=True,  # Set to True to enable summarization
        generate_audio=True  # Set to True to enable audio generation
    )

