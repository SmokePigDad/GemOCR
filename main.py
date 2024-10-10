import os
import time
from pathlib import Path
import google.generativeai as genai
from pdf2image import convert_from_path
import markdown
import argparse
import tkinter as tk
from tkinter import filedialog
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Try to get the API key from the environment, first from .env then from system environment variables
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")


def convert_pdf_to_images(pdf_path, output_folder):
    """Convert PDF to images."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i+1}.png')
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    
    return image_paths

def process_image(image_path):
    """Process a single image using Gemini API."""
    myfile = genai.upload_file(image_path)
    model = genai.GenerativeModel("gemini-1.5-flash-002")
    prompt = "Extract and transcribe all text from this image, preserving formatting where possible."
    response = model.generate_content([myfile, prompt])
    
    # Check if the response has a 'text' attribute
    if hasattr(response, 'text'):
        return response.text
    # If 'text' is not available, try to access the content directly
    elif hasattr(response, 'parts'):
        return ''.join(part.text for part in response.parts)
    else:
        raise ValueError("Unexpected response format from Gemini API")

def api_call_with_retry(func, max_retries=3):
    """Retry API call with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

def process_multiple_images(image_paths):
    """Process multiple images with error handling and retries."""
    results = []
    for path in image_paths:
        try:
            text = api_call_with_retry(lambda: process_image(path))
            results.append(text)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    return results

def compile_markdown(extracted_texts):
    """Compile extracted texts into a single Markdown string."""
    return "\n\n".join(extracted_texts)

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def create_pdf_with_text(texts, output_pdf_path):
    """Create a PDF with extracted text on corresponding pages, including word wrapping."""
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for text in texts:
        para = Paragraph(text, styles['Normal'])
        story.append(para)

    doc.build(story)

def pdf_to_markdown_and_pdf(pdf_path, output_markdown_path, pbar):
    """Convert PDF to Markdown and create a new PDF with extracted text."""
    try:
        # Step 1: Convert PDF to images
        temp_folder = 'temp_images'
        image_paths = convert_pdf_to_images(pdf_path, temp_folder)
        total_images = len(image_paths)

        # Step 2: Process images and extract text
        extracted_texts = []
        with tqdm.tqdm(total=total_images, desc="Processing Images", unit="image", leave=False) as image_pbar:
            for image_path in image_paths:
                try:
                    text = api_call_with_retry(lambda: process_image(image_path))
                    extracted_texts.append(text)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                finally:
                    image_pbar.update(1)
                    pbar.set_postfix({"image": image_pbar.n}) # Update main bar with image progress

        # Step 3: Compile Markdown
        markdown_content = compile_markdown(extracted_texts)

        # Step 4: Save Markdown file
        with open(output_markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Step 5: Create PDF with extracted text
        output_pdf_path = output_markdown_path.rsplit('.', 1)[0] + '_extracted.pdf'
        create_pdf_with_text(extracted_texts, output_pdf_path)

        print(f"Conversion complete. Markdown saved to {output_markdown_path}")
        print(f"Extracted text PDF saved to {output_pdf_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Clean up temporary images
        for path in image_paths:
            os.remove(path)
        os.rmdir(temp_folder)

def gui_pdf_to_markdown():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        print("No file selected. Exiting.")
        return

    # Create 'Output' folder if it doesn't exist
    output_folder = Path(os.path.dirname(os.path.abspath(__file__))) / "Output"
    output_folder.mkdir(exist_ok=True)

    # Generate output markdown filename
    pdf_name = Path(pdf_path).stem
    output_markdown_path = output_folder / f"{pdf_name}.md"

    pdf_to_markdown_and_pdf(pdf_path, str(output_markdown_path))

import shutil
import tqdm

def main():
    input_folder = "Input"
    processed_folder = "Processed"

    # Create folders if they don't exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]
    total_files = len(pdf_files)

    with tqdm.tqdm(total=total_files, desc="Processing PDFs", unit="file") as pbar:
        for filename in pdf_files:
            pdf_path = os.path.join(input_folder, filename)
            output_markdown_path = os.path.join(input_folder, filename[:-4] + ".md")

            try:
                pdf_to_markdown_and_pdf(pdf_path, output_markdown_path, pbar)  # Pass pbar to update progress

                processed_pdf_path = os.path.join(processed_folder, filename)
                shutil.move(pdf_path, processed_pdf_path)

            except Exception as e:
                print(f"An error occurred processing {filename}: {str(e)}")
            finally:
                pbar.update(1)


if __name__ == "__main__":
    main()
