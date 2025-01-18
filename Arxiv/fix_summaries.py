#!/usr/bin/env python3

"""
fix_summaries.py

This script processes all text files in the './summaries' directory, using a language model to convert lengthy text extracts into concise and coherent summaries. The revised texts are saved in a separate folder called './fixed_summaries'. Ensure you have the required package by installing it with 'pip install llama_index'.
"""

import os
from llama_index.llms.ollama import Ollama

def load_text_from_file(file_path: str) -> str:
    """Load raw text from a specified file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_text_to_file(file_path: str, text: str):
    """Write text to a specified file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

def rewrite_text_for_coherence(text: str) -> str:
    """
    Uses an LLM to rewrite 'text' in a more coherent and concise manner,
    excluding references to figures or tables not present in the text.
    """
    instruction_prompt = (
        "Rewrite the following text as a single coherent paragraph summarizing the research work. "
        "Use third person perspective (they, the authors, the researchers, etc.) rather than first person. "
        "Remove any meta-text like 'Here is a summary' or 'In this paper'. "
        "Eliminate references to figures, tables, or other non-text elements. "
        "Focus on describing what was done, found, and concluded in a natural flowing narrative. "
        "The output should be a single detailed paragraph without bullet points or lists."
    )
    prompt = f"{instruction_prompt}\n\nText:\n{text}"
    llm = Ollama(model="hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:latest", request_timeout=500.0)
    response = llm.complete(prompt)
    coherent_text = response.text.strip()
    return coherent_text

def main():
    input_folder = "./summaries"
    output_folder = "./fixed_summaries"
    
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_folder)
    
    input_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    output_files = set(os.listdir(output_folder))
    
    for filename in input_files:
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)
        
        if filename not in output_files:
            raw_text = load_text_from_file(input_file_path)
            revised_text = rewrite_text_for_coherence(raw_text)
            new_output_file_path = os.path.join(output_folder, f"fixed_{filename}")
            write_text_to_file(new_output_file_path, revised_text)
            print(f"Rewritten text has been saved to {new_output_file_path}")

if __name__ == "__main__":
    main()
