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
import shutil
import tqdm
from concurrent.futures import ProcessPoolExecutor
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime, timedelta

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

# Rate limiting parameters
requests_per_minute = 15
request_log = [] # Log of requests and their timestamps

def rate_limit():
    global request_log

    current_time = datetime.now()
    request_log = [req for req in request_log if current_time - req < timedelta(minutes=1)] #remove old requests
    if len(request_log) >= requests_per_minute:
        sleep_time = (datetime.now() - request_log[0] + timedelta(minutes=1)).total_seconds()
        print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
    request_log.append(current_time)


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
    """Process a single image using Gemini API with retry mechanism."""
    if not image_path:
        raise ValueError("image_path must be provided as a keyword argument.")

    max_retries = 3
    retries = 0
    while retries < max_retries:
        rate_limit()
        try:
            myfile = genai.upload_file(image_path)
            model = genai.GenerativeModel("gemini-1.5-flash-002")
            prompt = "Extract and transcribe all text from this image, preserving formatting where possible."
            response = model.generate_content([myfile, prompt])
            if hasattr(response, 'text') and response.text:
                return response.text
            elif hasattr(response, 'parts') and response.parts:
                extracted_text = ''.join(part.text for part in response.parts)
                if extracted_text:
                    return extracted_text
                else:
                    logging.error(f"No text extracted from image {image_path}. Response: {response}")
                    return None
            else:
                error_message = f"Unexpected response format from Gemini API: {response}"
                if hasattr(response, 'finish_reason'):
                    error_message += f", finish_reason: {response.finish_reason}"
                if hasattr(response, 'prompt_feedback'):
                    error_message += f", prompt_feedback: {response.prompt_feedback}"
                logging.error(error_message)
                raise ValueError(error_message)

        except genai.exception.GenerativeAIError as e:
            if e.code == 429:  # Rate limit exceeded
                sleep_time = 60  # Wait for 60 seconds
                print(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
                retries += 1
                continue  # Retry the request
            else:
                print(f"Error processing {image_path}: {e}")
                if os.path.exists(image_path):
                    os.remove(image_path)
                raise  # Re-raise other errors

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            if os.path.exists(image_path):
                os.remove(image_path)
            raise
    print(f"Failed to process {image_path} after multiple retries.")
    return None


def compile_markdown(extracted_texts):
    """Compile extracted texts into a single Markdown string."""
    return "\n\n".join(extracted_texts)


def create_pdf_with_text(texts, output_pdf_path):
    """Create a PDF with extracted text on corresponding pages, including word wrapping."""
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for text in texts:
        para = Paragraph(text, styles['Normal'])
        story.append(para)

    doc.build(story)


def pdf_to_markdown_and_pdf(pdf_path, output_markdown_path, output_pdf_path, pbar, thread_count=None):
    """Convert PDF to Markdown and create a new PDF with extracted text."""
    try:
        # Step 1: Convert PDF to images
        temp_folder = 'temp_images'
        image_paths = convert_pdf_to_images(pdf_path, temp_folder)
        total_images = len(image_paths)

        # Step 2: Process images and extract text
        # Use a ProcessPoolExecutor to process images concurrently
        if thread_count is None:
            thread_count = os.cpu_count() or 1  # Default to number of CPUs or 1 if not available

        with ProcessPoolExecutor(max_workers=thread_count) as executor:
            results = list(tqdm.tqdm(executor.map(process_image, image_paths),
                                      total=len(image_paths), desc="Processing Images", unit="image", leave=False))
        extracted_texts = [text for text in results if text is not None]  # Filter out None results

        # Step 3: Compile Markdown
        markdown_content = compile_markdown(extracted_texts)

        # Step 4: Save Markdown file
        with open(output_markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Step 5: Create PDF with extracted text
        #output_pdf_path = output_markdown_path.rsplit('.', 1)[0] + '.pdf' #This line is no longer needed

        print(f"Conversion complete. Markdown saved to {output_markdown_path}")


    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Clean up temporary images
        try:
            if os.path.exists(temp_folder):  # Check if the folder exists before attempting to remove it
                shutil.rmtree(temp_folder)
        except Exception as e:
            print(f"An error occurred during cleanup: {str(e)}")

    return extracted_texts  # Return extracted texts


def main():
    input_folder = "Input"
    processed_folder = "Processed"
    output_folder = "Output"

    # Create folders if they don't exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]
    total_files = len(pdf_files)

    with tqdm.tqdm(total=total_files, desc="Processing PDFs", unit="file") as pbar:
        for filename in pdf_files:
            pdf_path = os.path.join(input_folder, filename)
            output_filename = filename[:-4]  # remove '.pdf'
            output_markdown_path = os.path.join(output_folder, output_filename + ".md")
            output_pdf_path = os.path.join(output_folder, output_filename + ".pdf")

            try:
                extracted_texts = pdf_to_markdown_and_pdf(pdf_path, output_markdown_path, output_pdf_path, pbar, thread_count=4)  # Pass pbar and thread_count to update progress
                create_pdf_with_text(extracted_texts, output_pdf_path)  # create PDF in Output folder now that extracted_texts is available
                processed_pdf_path = os.path.join(processed_folder, filename)
                shutil.move(pdf_path, processed_pdf_path)


            except Exception as e:
                print(f"An error occurred processing {filename}: {str(e)}")
            finally:
                pbar.update(1)


if __name__ == "__main__":
    main()
