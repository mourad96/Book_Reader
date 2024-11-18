import os
import logging
import pytesseract
from PIL import Image
import fitz

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
                    image_path = os.path.join(output_dir, f"{page_number + 1}.png")
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