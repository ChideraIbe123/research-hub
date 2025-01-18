import os
import logging
from pathlib import Path
import re
import numpy as np
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, pipeline
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", batch_size: int = 4):
        """Initialize the summarizer with BART model and configurable batch size"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Initialize tokenizer and model only once
        logger.info(f"Loading model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=self.tokenizer,
            device=0 if str(self.device) == "cuda" else -1,
            framework="pt",
            batch_size=batch_size
        )
        
        # Download NLTK resources if needed
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        
        # Configure PDF extraction parameters
        self.laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5
        )

    @staticmethod
    def _ensure_nltk_resources():
        """Ensure required NLTK resources are available"""
        for resource in ['punkt', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF with error handling"""
        try:
            return extract_text(pdf_path, laparams=self.laparams)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return None

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text more efficiently"""
        if not text:
            return ""
            
        # Combine regex operations for better performance
        text = re.sub(
            r'(?:\s+)|(?:[^\w\s.,!?;:\-()])|(?:\f)|(?:\r)|(?:\s+\d+\s+)',
            lambda m: ' ' if m.group(0).isspace() else '',
            text
        )
        return text.strip()

    def get_important_sentences(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important sentences using TF-IDF with better error handling"""
        sentences = sent_tokenize(text)
        if len(sentences) <= top_n:
            return sentences
            
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            importance_scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
            top_indices = importance_scores.argsort()[-top_n:][::-1]
            return [sentences[i] for i in sorted(top_indices)]
        except Exception as e:
            logger.warning(f"TF-IDF processing failed: {e}. Falling back to first {top_n} sentences.")
            return sentences[:top_n]

    def create_smart_chunks(self, text: str, max_chunk_size: int = 1024) -> List[str]:
        """Create optimized chunks based on token length"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Pre-compute token lengths for all sentences
        sentence_lengths = [
            len(self.tokenizer.encode(sent)) for sent in sentences
        ]
        
        for sentence, length in zip(sentences, sentence_lengths):
            if current_length + length > max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = length
            else:
                current_chunk.append(sentence)
                current_length += length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def generate_chunk_summaries(self, chunks: List[str]) -> List[str]:
        """Generate summaries for chunks in parallel"""
        try:
            summaries = self.summarizer(
                chunks,
                max_length=300,
                min_length=100,
                do_sample=True,
                temperature=0.7,
                num_beams=4,
                batch_size=self.batch_size
            )
            return [summary['summary_text'] for summary in summaries]
        except Exception as e:
            logger.error(f"Error in batch summarization: {e}")
            return []

    def combine_summaries(self, summaries: List[str], cleaned_text: str) -> str:
        """Create an optimized combined summary"""
        if not summaries:
            return ""
            
        important_sentences = self.get_important_sentences(cleaned_text, top_n=15)
        combined_text = " ".join(summaries + important_sentences)
        
        try:
            prompts = [
                "Summarize the main points: " + combined_text,
                "Provide more specific details: " + combined_text,
                "Discuss implications and conclusions: " + combined_text
            ]
            
            results = self.summarizer(
                prompts,
                max_length=250,
                min_length=200,
                do_sample=True,
                temperature=0.7,
                num_beams=4,
                batch_size=3
            )
            
            return "\n\n".join(result['summary_text'] for result in results)
            
        except Exception as e:
            logger.error(f"Error in final summarization: {e}")
            return " ".join(summaries)

    def summarize_pdf(self, pdf_path: str) -> str:
        """Optimized main method to summarize a PDF"""
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            return ""
        
        cleaned_text = self.preprocess_text(raw_text)
        chunks = self.create_smart_chunks(cleaned_text)
        
        if not chunks:
            return ""
        
        chunk_summaries = self.generate_chunk_summaries(chunks)
        if not chunk_summaries:
            return ""
        
        final_summary = self.combine_summaries(chunk_summaries, cleaned_text)
        
        return final_summary

def main(source_folder: str = "arxiv_pdfs", destination_folder: str = "summaries"):
    logger.info("Initializing BART summarizer...")
    summarizer = PDFSummarizer()
    
    dest_path = Path(destination_folder)
    if dest_path.exists():
        for filename in os.listdir(dest_path):
            file_path = os.path.join(dest_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        dest_path.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_folder)
    pdf_files = list(source_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF file(s) in '{source_folder}'")
    
    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        
        summary = summarizer.summarize_pdf(str(pdf_path))
        
        if summary:
            output_path = dest_path / f"{pdf_path.stem}_summary.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary)
            
            logger.info(f"Summary saved to: {output_path}")
        else:
            logger.error(f"Failed to generate summary for: {pdf_path.name}")

if __name__ == "__main__":
    main()