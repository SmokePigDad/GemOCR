import os
import time
import logging
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

# Configure logging
logging.basicConfig(filename='gemocr.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

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
    image_path = kwargs.get("image_path") # Retrieve image_path from kwargs
    if not image_path:
        raise ValueError("image_path must be provided as a keyword argument.")

    try:
        myfile = genai.upload_file(image_path)
        model = genai.GenerativeModel("gemini-1.5-flash-002")
        prompt = "Extract and transcribe all text from this image, preserving formatting where possible."
        response = model.generate_content([myfile, prompt])
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        if os.path.exists(image_path):
            os.remove(image_path) # Clean up on error
        raise # Re-raise the exception after cleanup
    
    # Check if the response has a 'text' attribute
    if hasattr(response, 'text') and response.text: # Check for text and if it's not empty
        return response.text
    # If 'text' is not available or empty, try to access the content directly
    elif hasattr(response, 'parts') and response.parts: # Check for parts and if it's not empty
        extracted_text = ''.join(part.text for part in response.parts)
        if extracted_text: # Check if extracted text is not empty
            return extracted_text
        else:
            logging.error(f"No text extracted from image {image_path}. Response: {response}")
            return None # Return None to indicate failure
    else:
        error_message = f"Unexpected response format from Gemini API: {response}"
        if hasattr(response, 'finish_reason'):
            error_message += f", finish_reason: {response.finish_reason}"
        if hasattr(response, 'prompt_feedback'):
            error_message += f", prompt_feedback: {response.prompt_feedback}"
        logging.error(error_message)
        raise ValueError(error_message)

def api_call_with_retry(func, max_retries=3):
    """Retry API call with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Max retries reached for {func.__name__}. Raising exception.") # More informative logging
                raise
            print(f"Retrying {func.__name__} after exception: {e}") # Log retry attempts
            time.sleep(2 ** attempt * 0.5 + (attempt * 0.1)) # Add some jitter to avoid synchronized retries

def api_call_with_retry(func, max_retries=5, retry_exceptions=(Exception,), *args, **kwargs): # Add *args and **kwargs
    """Retry API call with exponential backoff and jitter."""
    for path in image_paths:
        text = api_call_with_retry(process_image, max_retries=5, retry_exceptions=(Exception,), image_path=path) # Pass path as keyword argument
        if text: # Only append if text was extracted
            results.append(text)
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

def pdf_to_markdown_and_pdf(pdf_path, output_markdown_path, output_pdf_path, pbar):
    """Convert PDF to Markdown and create a new PDF with extracted text."""
    try:
        # Step 1: Convert PDF to images
        temp_folder = 'temp_images'
        image_paths = convert_pdf_to_images(pdf_path, temp_folder)
        total_images = len(image_paths)

        # Step 2: Process images and extract text
        # Use a ThreadPoolExecutor to process images concurrently
        with ThreadPoolExecutor() as executor:
            results = list(tqdm.tqdm(executor.map(api_call_with_retry, [lambda: process_image(path) for path in image_paths]),
                                      total=len(image_paths), desc="Processing Images", unit="image", leave=False))
        extracted_texts = [text for text in results if text is not None] # Filter out None results

        # Step 3: Compile Markdown
        markdown_content = compile_markdown(extracted_texts)

        # Step 4: Save Markdown file
        with open(output_markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Step 5: Create PDF with extracted text
        output_pdf_path = output_markdown_path.rsplit('.', 1)[0] + '.pdf'
        # Step 5: Create PDF with extracted text - MOVED to main function

        print(f"Conversion complete. Markdown saved to {output_markdown_path}")


    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary images - moved inside finally block
        if os.path.exists(temp_folder): # Check if the folder exists before attempting to remove it
            for filename in os.listdir(temp_folder):
                file_path = os.path.join(temp_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            try:
                os.rmdir(temp_folder)
            except OSError as e:
                print(f"Error removing directory {temp_folder}: {e}")

    return extracted_texts # Return extracted texts

import shutil
import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
            output_filename = filename[:-4] # remove '.pdf'
            output_markdown_path = os.path.join("Output", output_filename + ".md")

            try:
                output_pdf_path = os.path.join("Output", output_filename + ".pdf") # construct output PDF path
                extracted_texts = pdf_to_markdown_and_pdf(pdf_path, output_markdown_path, output_pdf_path, pbar)  # Pass pbar to update progress
                create_pdf_with_text(extracted_texts, output_pdf_path) # create PDF in Output folder now that extracted_texts is available
                processed_pdf_path = os.path.join(processed_folder, filename)
                shutil.move(pdf_path, processed_pdf_path)


            except Exception as e:
                print(f"An error occurred processing {filename}: {str(e)}")
            finally:
                pbar.update(1)


if __name__ == "__main__":
    main()
