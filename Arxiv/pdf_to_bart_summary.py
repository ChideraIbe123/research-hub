import os
import re
import math
import logging
from pathlib import Path
from pdfminer.high_level import extract_text
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
import torch
import time

def setup_logging(log_file='bart_summary.log'):
    """Set up logging to file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # File handler for detailed logs
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler for general info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

def sanitize_filename(filename):
    """Remove or replace characters that are invalid in filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def extract_pdf_text(pdf_path):
    """Extract text from a single PDF file."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def split_text_into_chunks(text, max_tokens=1024, tokenizer=None, min_words=30):
    """
    Split text into chunks based on the tokenizer's tokenization.
    Ensures each chunk has at most `max_tokens` tokens and at least `min_words` words.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided.")

    tokens = tokenizer.encode(text, return_tensors='pt')[0]
    total_tokens = len(tokens)
    chunks = []
    for i in range(0, total_tokens, max_tokens):
        chunk = tokens[i:i + max_tokens]
        # Ensure chunk is not empty
        if len(chunk) == 0:
            continue
        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True).strip()
        # Skip chunks that are too short to summarize
        if len(decoded_chunk.split()) < min_words:
            logging.debug(f"Skipping chunk {len(chunks)+1} due to insufficient length.")
            continue
        chunks.append(decoded_chunk)
    return chunks

def summarize_chunks(chunks, summarizer, tokenizer, device, max_retries=3, backoff_factor=2):
    """Summarize each chunk using the provided summarizer pipeline with retry logic."""
    summaries = []
    for idx, chunk in enumerate(chunks, start=1):
        # Calculate input length
        input_length = len(tokenizer.encode(chunk, return_tensors='pt')[0])
        # Adjust max_length to be less than input_length
        adjusted_max_length = min(300, input_length - 10) if input_length > 80 else 80

        if adjusted_max_length < 80:
            logging.warning(f"Chunk {idx} is too short for summarization. Skipping.")
            summaries.append("[Summary not available due to insufficient length]")
            continue

        attempt = 0
        while attempt < max_retries:
            try:
                logging.debug(f"Summarizing chunk {idx}/{len(chunks)} (Attempt {attempt + 1})...")
                summary = summarizer(
                    chunk, 
                    max_length=adjusted_max_length, 
                    min_length=80, 
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
                break  # Exit the retry loop on success
            except Exception as e:
                attempt += 1
                logging.error(f"Error summarizing chunk {idx} on attempt {attempt}: {e}")
                if attempt < max_retries:
                    sleep_time = backoff_factor ** attempt
                    logging.info(f"Retrying summarization for chunk {idx} in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Failed to summarize chunk {idx} after {max_retries} attempts.")
                    summaries.append("[Summary not available]")
    return summaries

def aggregate_summaries(summaries):
    """Aggregate individual summaries into a single summary."""
    combined_summary = " ".join(summaries)
    return combined_summary

def process_pdfs(source_folder="arxiv_pdfs", destination_folder="summaries"):
    """
    Process all PDFs in the source_folder:
    - Extract text
    - Split into chunks
    - Summarize each chunk using BART
    - Save each summary to a different file in the destination_folder
    """
    # Ensure the source_folder is a full path to avoid issues with relative paths
    source_path = Path(os.path.abspath(source_folder))
    dest_path = Path(os.path.abspath(destination_folder))

    # Verify source folder
    if not source_path.exists() or not source_path.is_dir():
        logging.error(f"Source folder '{source_folder}' does not exist or is not a directory.")
        return

    # Create destination folder
    dest_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Destination folder set to '{destination_folder}'.")

    # Initialize BART summarizer
    logging.info("Loading BART summarization model...")
    device = 0 if torch.cuda.is_available() else -1
    try:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer=tokenizer, device=device)
    except Exception as e:
        logging.error(f"Error loading BART model or tokenizer: {e}")
        return

    # Iterate over PDFs
    pdf_files = list(source_path.glob("*.pdf"))
    if not pdf_files:
        logging.warning(f"No PDF files found in '{source_folder}'. Exiting.")
        return

    logging.info(f"Found {len(pdf_files)} PDF file(s) in '{source_folder}'. Starting processing...")

    for pdf_file in pdf_files:
        logging.info(f"Processing '{pdf_file.name}'...")
        text = extract_pdf_text(pdf_file)
        if not text.strip():
            logging.warning(f"No text extracted from '{pdf_file.name}'. Skipping.")
            continue

        # Optional: Clean the text (remove multiple spaces, etc.)
        text = re.sub(r'\s+', ' ', text)

        # Split text into chunks
        chunks = split_text_into_chunks(text, max_tokens=1024, tokenizer=tokenizer, min_words=30)
        logging.info(f"Extracted {len(chunks)} chunk(s) from '{pdf_file.name}'.")

        if not chunks:
            logging.warning(f"No suitable chunks extracted from '{pdf_file.name}'. Skipping summarization.")
            continue

        # Summarize each chunk
        summaries = summarize_chunks(chunks, summarizer, tokenizer, device)

        # Filter out "[Summary not available]" from summaries
        summaries = [summary for summary in summaries if summary != "[Summary not available]"]

        # Aggregate summaries
        aggregated_summary = aggregate_summaries(summaries)

        # Further summarize the aggregated summary if it's too long
        aggregated_length = len(tokenizer.encode(aggregated_summary, return_tensors='pt')[0])
        if aggregated_length > 2048:
            logging.info(f"Aggregated summary too long for BART. Summarizing again.")
            try:
                final_summary = summarizer(
                    aggregated_summary, 
                    max_length=1024,  # Increase max_length to ensure longer summaries
                    min_length=300,   # Increase min_length to ensure longer summaries
                    do_sample=False
                )[0]['summary_text']
                aggregated_summary = final_summary
            except Exception as e:
                logging.error(f"Error during final summarization: {e}")
                aggregated_summary = "[Final summary not available]"

        # Define output file path
        base_name = pdf_file.stem
        safe_name = sanitize_filename(base_name)
        output_txt = dest_path / f"{safe_name}_summary.txt"

        # Save the summary
        try:
            with open(output_txt, "w", encoding="utf-8") as f_out:
                f_out.write(aggregated_summary)
            logging.info(f"Saved summary to '{output_txt.name}'.")
        except Exception as e:
            logging.error(f"Error saving summary for '{pdf_file.name}': {e}")

if __name__ == "__main__":
    setup_logging()
    process_pdfs("arxiv_pdfs", "summaries")
