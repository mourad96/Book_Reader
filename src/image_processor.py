import os
import logging
from google import genai
from google.genai import types
from PIL import Image
import fitz  # PyMuPDF
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import io

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageProcessor:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def convert_pdf_to_cropped_images(self, pdf_path, output_dir, start_page=1, end_page=None, dpi=300):
        """Convert PDF to cropped images and detect footer lines dynamically."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
            
        try:
            with fitz.open(pdf_path) as pdf:
                total_pages = len(pdf)
                end_page = end_page or total_pages

                if start_page < 1 or end_page > total_pages or start_page > end_page:
                    raise ValueError("Invalid page range specified.")

                for page_number in range(start_page - 1, end_page):
                    page = pdf[page_number]
                    pix = page.get_pixmap(dpi=dpi)

                    # Save the image temporarily
                    full_image_path = os.path.join(output_dir, f"{page_number + 1}_full.png")
                    pix.save(full_image_path)

                    # Process the image to crop below the footer line
                    cropped_image = self.detect_and_crop_footer(full_image_path)
                    cropped_image_path = os.path.join(output_dir, f"{page_number + 1}.png")
                    cropped_image.save(cropped_image_path)
                    logging.info(f"Saved cropped image for page {page_number + 1} to {cropped_image_path}")

                    # Remove the full image file after cropping
                    os.remove(full_image_path)

        except Exception as e:
            logging.error(f"Failed to convert PDF to cropped images: {e}")
            raise

    def detect_and_crop_footer(self, image_path):
        """Detect and crop the footer based on advanced footer line detection for book pages."""
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Could not read image from {image_path}")
            # Return original image from path if cv2 fails
            return Image.open(image_path)
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(binary, 30, 200)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        lines = cv2.HoughLinesP(dilated, rho=1, theta=np.pi/180, threshold=50, 
                              minLineLength=width//6, maxLineGap=10)

        potential_footer_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10:  # Horizontal line check
                    line_length = abs(x2 - x1)
                    line_position = (y1 + y2) / 2
                    if (line_length > 0.2 * width and 
                        line_position > 0.67 * height and 
                        line_position < 0.95 * height):
                        potential_footer_lines.append((line_position, line_length))

        footer_line_y = None
        if potential_footer_lines:
            potential_footer_lines.sort(key=lambda x: x[1], reverse=True)
            footer_line_y = int(potential_footer_lines[0][0])
            logging.info(f"Footer line detected at y={footer_line_y}")
        else:
            footer_line_y = int(0.9 * height)
            logging.info("No strong footer line detected. Using default crop position.")

        crop_region = (0, 0, width, footer_line_y)
        cropped_image = Image.open(image_path).crop(crop_region)
        return cropped_image

    @staticmethod
    def get_sorted_image_files(directory):
        """Fetch and sort image files by numeric prefix."""
        valid_extensions = (".png", ".jpg", ".jpeg")
        try:
            files = [filename for filename in os.listdir(directory) if filename.endswith(valid_extensions)]
        except FileNotFoundError:
            logging.error(f"Directory not found: {directory}")
            return []

        def extract_numeric_prefix(filename):
            try:
                return int(filename.split('.')[0])
            except (ValueError, IndexError):
                return float('inf')

        sorted_files = sorted(files, key=extract_numeric_prefix)
        return [os.path.join(directory, filename) for filename in sorted_files]

    def extract_text_from_image(self, image_path, lang='ara'):
        """Extract text from an image using Google Gemini 2.5 Flash."""
        logging.info(f"Processing image for text extraction: {image_path}")
        try:
            # Load the image
            img = Image.open(image_path)
            
            # Convert PIL Image to bytes for the API
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Create a more specific prompt for better results
            if lang == 'ara':
                prompt = "Extract all Arabic text from this image. Preserve the original text structure and formatting as much as possible. Output only the extracted text without any additional commentary."
            else:
                prompt = f"Extract all text from this image. The text is in {lang} language. Preserve the original text structure and formatting as much as possible. Output only the extracted text without any additional commentary."

            # Generate content with the image and prompt using the new API
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(
                            mime_type="image/png",
                            data=img_bytes.getvalue()
                        ))
                    ])
                ]
            )
            
            # Check if response was successful
            if response.candidates and response.candidates[0].content.parts:
                text = response.candidates[0].content.parts[0].text
                return text.strip() if text else ""
            else:
                logging.warning(f"No text returned for {image_path}")
                # Check for safety ratings or other blocking reasons
                if response.candidates and response.candidates[0].finish_reason:
                    logging.warning(f"Finish reason: {response.candidates[0].finish_reason}")
                if response.candidates and response.candidates[0].safety_ratings:
                    logging.warning(f"Safety ratings: {response.candidates[0].safety_ratings}")
                return ""
            
        except Exception as e:
            logging.error(f"Failed to process {image_path} with Gemini: {e}")
            return ""

    def extract_text_from_images(self, image_files, lang='ara'):
        """Extract text from a list of sorted image files and measure execution time."""
        if not image_files:
            logging.warning("No image files to process.")
            return ""
            
        start_time = time.time()

        # Process images with threading for better performance
        # Note: Be careful with rate limits when using threading with API calls
        with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced workers to respect rate limits
            results = list(executor.map(lambda img: self.extract_text_from_image(img, lang), image_files))
        
        # Join results with proper spacing
        text = "\n".join(filter(None, results))  # Filter out empty results

        end_time = time.time()
        logging.info(f"Execution time for extracting text from {len(image_files)} images: {end_time - start_time:.2f} seconds")

        return text

    def extract_text_from_pdf(self, pdf_path, output_dir, start_page=1, end_page=None, lang='ara', extract_text=True):
        """Convert a range of pages from a PDF to cropped images and optionally extract text."""
        self.convert_pdf_to_cropped_images(pdf_path, output_dir, start_page, end_page)
        
        if extract_text:
            sorted_images = self.get_sorted_image_files(output_dir)
            return self.extract_text_from_images(sorted_images, lang)
        return None