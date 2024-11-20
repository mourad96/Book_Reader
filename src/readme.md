# Image Processor Module

This module provides functionality for processing PDFs and extracting text from images, with advanced footer detection and cropping features.

## Features
- Convert PDF pages to cropped images.
- Detect and crop footer regions dynamically using Hough Line Transform and adaptive thresholding.
- Extract text from images using Tesseract OCR.
- Process a range of PDF pages and save the output in a specified directory.

## Installation
1. Ensure the correct installation of dependencies:
   - `pytesseract` with a valid Tesseract-OCR installation.
   - `PyMuPDF` (`fitz`), `Pillow`, and `OpenCV`.

2. Configure `tesseract_cmd` to point to your Tesseract installation path.

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

