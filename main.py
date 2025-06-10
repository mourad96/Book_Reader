import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.image_processor import ImageProcessor
from src.text_processor import TextProcessor
from src.text_to_speech import TextToSpeech

def main(file_dir, output_text_file, summary_file, audio_file, api_key, external_user_id,
         start_page=None, end_page=None, summarize=False, generate_audio=False, generate_img=True, extract_txt=False):
    logging.basicConfig(level=logging.INFO)
    processor = ImageProcessor(api_key)
    text_processor = TextProcessor(api_key, external_user_id)
    synthesizer = TextToSpeech(use_google=True) #change it to false if you want to use offline gtts

    # Ensure file_dir contains only one PDF to process
    pdf_files = [f for f in os.listdir(file_dir) if f.endswith('.pdf')]
    if len(pdf_files) != 1:
        raise ValueError("Please ensure the directory contains exactly one PDF to process.")

    pdf_path = os.path.join(file_dir, pdf_files[0])
    output_dir = "./images"
    os.makedirs(output_dir, exist_ok=True)

    # Handle image generation independently
    if generate_img:
        logging.info(f"Processing PDF: {pdf_path}, Pages: {start_page}-{end_page}")
        processor.extract_text_from_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            start_page=start_page,
            end_page=end_page,
            extract_text=False  # Only generate images without text extraction
        )
        logging.info("Image generation completed")

    # Initialize extracted_text
    extracted_text = None

    if extract_txt:
        # Extract text from images
        logging.info("Extracting text from images...")
        sorted_images = processor.get_sorted_image_files(output_dir)
        extracted_text = processor.extract_text_from_images(sorted_images)

        # Save the extracted text
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        logging.info(f"Extracted text saved to {output_text_file}")
    else:
        # Read the text directly from the existing output.txt file
        if os.path.exists(output_text_file):
            with open(output_text_file, "r", encoding="utf-8") as f:
                extracted_text = f.read()
            logging.info(f"Using text from existing file: {output_text_file}")
        else:
            if summarize or generate_audio:
                raise FileNotFoundError(f"The specified text file '{output_text_file}' does not exist.")

    # Summarize text if the option is enabled
    summarized_text = None
    if summarize and extracted_text:
        logging.info("Summarizing the extracted text...")
        summarized_text = text_processor.summarize_text(extracted_text)

        # Save the summarized text
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summarized_text)
        logging.info(f"Summarized text saved to {summary_file}")
    else:
        summarized_text = extracted_text

    # Convert text to audio if the option is enabled
    if generate_audio and summarized_text:
        synthesizer.text_to_audio(summarized_text, audio_file)
        logging.info(f"Audio file saved to {audio_file}")


if __name__ == "__main__":
    load_dotenv()
    
    main(
        file_dir="./",
        output_text_file="output.txt",
        summary_file="summary.txt",
        audio_file="output.mp3",
        api_key=os.getenv("GEMINI_API_KEY"),
        external_user_id=os.getenv("EXTERNAL_USER_ID"),
        start_page=11,
        end_page=16,
        summarize=False,
        generate_audio=False,
        generate_img=False,
        extract_txt=True
    )