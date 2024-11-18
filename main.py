import pytesseract
from PIL import Image
import os
from gtts import gTTS
from pathlib import Path
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

class ImageProcessor:
    def __init__(self, tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    @staticmethod
    def get_sorted_image_files(directory):
        """Fetch and sort image files by numeric prefix."""
        image_files = [filename for filename in os.listdir(directory) if filename.endswith((".jpg", ".png"))]
        sorted_image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))
        return [os.path.join(directory, filename) for filename in sorted_image_files]

    def extract_text(self, image_path, lang='ara'):
        """Extract text from a single image."""
        try:
            return pytesseract.image_to_string(Image.open(image_path), lang=lang, config="--psm 6")
        except Exception as e:
            logging.error(f"Failed to process {image_path}: {e}")
            return ""

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

def main(image_dir, output_text_file, summary_file, audio_file, tesseract_cmd, api_key, external_user_id, summarize=False, generate_audio=False):
    logging.basicConfig(level=logging.INFO)
    processor = ImageProcessor(tesseract_cmd)
    text_processor = TextProcessor(api_key, external_user_id)
    synthesizer = TextToSpeech()

    # Fetch and sort images
    sorted_image_paths = processor.get_sorted_image_files(image_dir)
    logging.info(f"Sorted image files: {sorted_image_paths}")

    # Extract text from images
    extracted_text = ""
    for img_path in sorted_image_paths:
        extracted_text += processor.extract_text(img_path) + "\n"

    # Save full text
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    logging.info(f"Extracted text saved to {output_text_file}")

    # Summarize text if the option is enabled
    if summarize:
        logging.info("Summarizing the extracted text...")
        summarized_text = text_processor.summarize_text(extracted_text)

        # Save summary to file
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summarized_text)
        logging.info(f"Summarized text saved to {summary_file}")
    else:
        summarized_text = extracted_text

    # Convert summarized or full text to speech if enabled
    if generate_audio:
        synthesizer.text_to_audio(summarized_text, audio_file)

if __name__ == "__main__":
    main(
        image_dir="./",
        output_text_file="output.txt",
        summary_file="summary.txt",
        audio_file="output.mp3",
        tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        api_key=os.getenv("API_KEY"),  # Replace with your actual API key
        external_user_id=os.getenv("EXTERNAL_USER_ID"),  # Replace with your actual external user ID
        summarize=True,  # Set to True to enable summarization
        generate_audio=False  # Set to True to enable audio generation
    )
