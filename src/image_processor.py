import os
import logging
import google.generativeai as genai
from PIL import Image
import fitz
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import base64
import io


class ImageProcessor:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def convert_pdf_to_cropped_images(self, pdf_path, output_dir, start_page=1, end_page=None, dpi=300):
        """Convert PDF to cropped images and detect footer lines dynamically."""
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

    def detect_and_crop_footer(self, image_path):
        """Detect and crop the footer based on advanced footer line detection for book pages."""
        # Load the image and convert to grayscale
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Apply adaptive thresholding to better handle varying page backgrounds
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(binary, 30, 200)

        # Apply dilation to connect nearby edges
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Detect horizontal lines using Hough Line Transform
        lines = cv2.HoughLinesP(dilated, rho=1, theta=np.pi/180, threshold=50, 
                               minLineLength=width//6, maxLineGap=10)

        # # Create a debug image to visualize detected lines
        # debug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        potential_footer_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is horizontal (allowing small slope)
                if abs(y2 - y1) < 10:
                    # Calculate line properties
                    line_length = abs(x2 - x1)
                    line_position = (y1 + y2) / 2
                    
                    # # Draw all horizontal lines in blue for debugging
                    # cv2.line(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Filter lines based on:
                    # 1. Length (at least 20% of page width)
                    # 2. Position (in bottom third but not at very bottom)
                    if (line_length > 0.2 * width and 
                        line_position > 0.67 * height and 
                        line_position < 0.95 * height):
                        potential_footer_lines.append((line_position, line_length))
                        # # Draw potential footer lines in green
                        # cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        footer_line_y = None
        if potential_footer_lines:
            # Select the longest line as the footer line
            potential_footer_lines.sort(key=lambda x: x[1], reverse=True)
            footer_line_y = int(potential_footer_lines[0][0])
            logging.info(f"Footer line detected at y={footer_line_y}")
            # # Draw the selected footer line in red
            # cv2.line(debug_image, (0, footer_line_y), (width, footer_line_y), (0, 0, 255), 3)
        else:
            # If still no line found, use conservative default
            if footer_line_y is None:
                footer_line_y = int(0.9 * height)
                logging.info("Using default footer line position")

            # # Draw default footer line in yellow
            # cv2.line(debug_image, (0, footer_line_y), (width, footer_line_y), (0, 255, 255), 3)

        # # Save debug image showing all detected lines
        # debug_image_path = image_path.replace('.png', '_debug_lines.png')
        # cv2.imwrite(debug_image_path, debug_image)
        # logging.info(f"Saved debug image showing detected lines to {debug_image_path}")

        # Crop the image above the detected footer line
        crop_region = (0, 0, width, footer_line_y)
        cropped_image = Image.open(image_path).crop(crop_region)
        return cropped_image

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

    def extract_text_from_image(self, image_path, lang='ara'):
        """Extract text from an image using Google Gemini Vision."""
        try:
            # Open and prepare the image
            with Image.open(image_path) as img:
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format or 'PNG')
                img_byte_arr = img_byte_arr.getvalue()

            # Create prompt for text extraction
            prompt = f"Extract all text from this image. The text is in {lang} language. Output only the extracted text, without any additional commentary."

            # Generate content using Gemini
            response = self.model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_byte_arr}
            ])

            # Get the extracted text
            return response.text.strip()
            
        except Exception as e:
            logging.error(f"Failed to process {image_path}: {e}")
            return ""

    def extract_text_from_images(self, image_files, lang='ara'):
        """Extract text from a list of sorted image files and measure execution time."""
        start_time = time.time()

        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda img: self.extract_text_from_image(img, lang), image_files)
        text = "".join(results)

        end_time = time.time()
        logging.info(f"Execution time for extract_text_from_images: {end_time - start_time:.2f} seconds")

        return text

    def extract_text_from_pdf(self, pdf_path, output_dir, start_page=1, end_page=None, lang='ara'):
        """Convert a range of pages from a PDF to cropped images and extract text."""
        self.convert_pdf_to_cropped_images(pdf_path, output_dir, start_page, end_page)
        sorted_images = self.get_sorted_image_files(output_dir)
        return self.extract_text_from_images(sorted_images, lang)
