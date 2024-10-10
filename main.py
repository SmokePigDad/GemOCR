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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime, timedelta
import random
import threading
import requests
import ssl
import urllib3
import socket

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('gemocr.log'),
                        logging.StreamHandler()
                    ])

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create a custom SSL context
custom_ssl_context = ssl.create_default_context()
custom_ssl_context.check_hostname = False
custom_ssl_context.verify_mode = ssl.CERT_NONE

# Load environment variables from .env file if it exists
load_dotenv()

# Try to get the API key from the environment, first from .env then from system environment variables
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

# Rate limiting parameters
requests_per_minute = 60  # Adjusted to match Gemini API limit
request_interval = 60 / requests_per_minute
last_request_time = 0
rate_limit_lock = threading.Lock()

def rate_limit():
    global last_request_time
    with rate_limit_lock:
        current_time = time.time()
        time_since_last_request = current_time - last_request_time
        if time_since_last_request < request_interval:
            sleep_time = request_interval - time_since_last_request
            time.sleep(sleep_time)
        last_request_time = time.time()

def exponential_backoff(attempt):
    return min(600, (2 ** attempt) + (random.randint(0, 1000) / 1000))

def is_rate_limit_error(e):
    return isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429


def convert_pdf_to_images(pdf_path, output_folder):
    """Convert PDF to images using a thread pool."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = convert_from_path(pdf_path)
    image_paths = []

    def save_image(args):
        i, image = args
        image_path = os.path.join(output_folder, f'page_{i+1}.png')
        image.save(image_path, 'PNG')
        return image_path

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        image_paths = list(executor.map(save_image, enumerate(images)))

    return image_paths


def process_image(image_path):
    """Process a single image using Gemini API with retry mechanism."""
    if not image_path:
        raise ValueError("image_path must be provided as a keyword argument.")

    max_retries = 10
    for attempt in range(max_retries):
        try:
            rate_limit()
            logging.info(f"Processing image: {image_path}")
            
            # Use custom SSL context for file upload
            original_context = ssl._create_default_https_context
            ssl._create_default_https_context = lambda: custom_ssl_context
            try:
                myfile = genai.upload_file(image_path)
            except (ssl.SSLError, socket.error) as e:
                logging.error(f"SSL or socket error during file upload for {image_path}: {e}")
                raise
            finally:
                ssl._create_default_https_context = original_context

            model = genai.GenerativeModel("gemini-1.5-flash-002")
            prompt = "Extract and transcribe all text from this image, preserving formatting where possible."
            
            # Disable SSL verification for content generation
            with requests.Session() as session:
                session.verify = False
                response = model.generate_content([myfile, prompt], timeout=300)

            if hasattr(response, 'text') and response.text:
                logging.info(f"Successfully processed image: {image_path}")
                return response.text
            elif hasattr(response, 'parts') and response.parts:
                extracted_text = ''.join(part.text for part in response.parts)
                if extracted_text:
                    logging.info(f"Successfully processed image: {image_path}")
                    return extracted_text
                else:
                    logging.warning(f"No text extracted from image {image_path}. Response: {response}")
                    return None
            else:
                error_message = f"Unexpected response format from Gemini API: {response}"
                if hasattr(response, 'finish_reason'):
                    error_message += f", finish_reason: {response.finish_reason}"
                if hasattr(response, 'prompt_feedback'):
                    error_message += f", prompt_feedback: {response.prompt_feedback}"
                logging.error(error_message)
                raise ValueError(error_message)

        except genai.types.generation_types.BlockedPromptException as e:
            logging.error(f"Blocked prompt exception for {image_path}: {e}")
            if os.path.exists(image_path):
                os.remove(image_path)
            raise
        except (requests.exceptions.RequestException, TimeoutError, ssl.SSLError) as e:
            if is_rate_limit_error(e):
                sleep_time = exponential_backoff(attempt)
                logging.warning(f"Rate limit reached. Retrying {image_path} in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            elif isinstance(e, ssl.SSLError):
                logging.error(f"SSL error processing {image_path}: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    sleep_time = exponential_backoff(attempt)
                    logging.info(f"Retrying {image_path} in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Failed to process {image_path} after {max_retries} attempts due to SSL errors.")
                    return None
            else:
                logging.error(f"Error processing {image_path}: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    sleep_time = exponential_backoff(attempt)
                    logging.info(f"Retrying {image_path} in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Failed to process {image_path} after {max_retries} attempts.")
                    return None
        except Exception as e:
            logging.error(f"Unexpected error processing {image_path}: {e}", exc_info=True)
            if attempt < max_retries - 1:
                sleep_time = exponential_backoff(attempt)
                logging.info(f"Retrying {image_path} in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"Failed to process {image_path} after {max_retries} attempts.")
                return None

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


def pdf_to_markdown_and_pdf(pdf_path, output_markdown_path, output_pdf_path, pbar):
    """Convert PDF to Markdown and create a new PDF with extracted text."""
    temp_folder = 'temp_images'
    try:
        logging.info(f"Starting conversion of {pdf_path}")
        
        # Step 1: Convert PDF to images
        logging.info(f"Converting PDF to images: {pdf_path}")
        image_paths = convert_pdf_to_images(pdf_path, temp_folder)
        total_images = len(image_paths)
        logging.info(f"Converted PDF {pdf_path} to {total_images} images")

        # Step 2: Process images and extract text
        logging.info(f"Processing {total_images} images")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_image, image_path) for image_path in image_paths]
            extracted_texts = []
            for i, future in enumerate(tqdm.tqdm(as_completed(futures), total=total_images, desc="Processing Images", unit="image", leave=False)):
                try:
                    result = future.result(timeout=360)  # 6 minutes timeout
                    if result:
                        extracted_texts.append(result)
                    else:
                        logging.warning(f"No text extracted from image {i+1}")
                    pbar.update(1 / total_images)  # Update progress bar
                except TimeoutError:
                    logging.error(f"Timeout occurred while processing image {i+1}")
                except Exception as e:
                    logging.error(f"Error processing image {i+1}: {str(e)}")

        if not extracted_texts:
            logging.error("No text was extracted from any of the images.")
            return None

        # Step 3: Compile Markdown
        logging.info("Compiling extracted text into Markdown")
        markdown_content = compile_markdown(extracted_texts)

        # Step 4: Save Markdown file
        logging.info(f"Saving Markdown to {output_markdown_path}")
        with open(output_markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Step 5: Create PDF with extracted text
        logging.info(f"Creating PDF with extracted text: {output_pdf_path}")
        create_pdf_with_text(extracted_texts, output_pdf_path)

        logging.info(f"Conversion complete. Markdown saved to {output_markdown_path}")
        return extracted_texts

    except Exception as e:
        logging.exception(f"An error occurred during PDF processing: {str(e)}")
        return None

    finally:
        # Clean up temporary images
        logging.info("Cleaning up temporary images")
        try:
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
            logging.info("Temporary images cleaned up successfully")
        except Exception as e:
            logging.error(f"An error occurred during cleanup: {str(e)}")



def main():
    logging.info("Script started")
    input_folder = "Input"
    processed_folder = "Processed"
    output_folder = "Output"

    # Create folders if they don't exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]
    total_files = len(pdf_files)

    logging.info(f"Found {total_files} PDF files in the input folder.")

    if total_files == 0:
        logging.warning("No PDF files found in the input folder. Exiting.")
        print("No PDF files found in the input folder. Please add PDF files and run the script again.")
        return

    try:
        with tqdm.tqdm(total=total_files, desc="Processing PDFs", unit="file") as pbar:
            for filename in pdf_files:
                pdf_path = os.path.join(input_folder, filename)
                output_filename = filename[:-4]  # remove '.pdf'
                output_markdown_path = os.path.join(output_folder, output_filename + ".md")
                output_pdf_path = os.path.join(output_folder, output_filename + ".pdf")

                logging.info(f"Starting to process {filename}")
                try:
                    extracted_texts = pdf_to_markdown_and_pdf(pdf_path, output_markdown_path, output_pdf_path, pbar)
                    if extracted_texts:
                        create_pdf_with_text(extracted_texts, output_pdf_path)
                        processed_pdf_path = os.path.join(processed_folder, filename)
                        shutil.move(pdf_path, processed_pdf_path)
                        logging.info(f"Successfully processed {filename}")
                    else:
                        logging.error(f"Failed to extract text from {filename}")
                        print(f"Failed to extract text from {filename}. Check the log file for details.")
                except Exception as e:
                    logging.exception(f"An error occurred processing {filename}: {str(e)}")
                    print(f"An error occurred processing {filename}. Check the log file for details.")
                finally:
                    pbar.update(1)

        logging.info("Processing completed. Check the log file for details.")
        print("Processing completed. Check the log file for details.")
    except Exception as e:
        logging.critical(f"A critical error occurred during execution: {str(e)}")
        print(f"A critical error occurred. Please check the log file for details.")

if __name__ == "__main__":
    main()
