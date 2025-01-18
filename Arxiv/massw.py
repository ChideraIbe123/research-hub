#!/usr/bin/env python3

"""
massw.py

This script takes an input text and uses a language model to predict the potential impact on the military. It provides up to three ways the input can be utilized to influence military operations or strategies.
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

def predict_military_impact(text: str) -> str:
    """
    Uses an LLM to predict the potential impact of the input text on the military,
    providing up to three ways it can be used.
    """
    instruction_prompt = (
        "Examine the following text and identify three potential military applications or impacts. "
        "For each application, provide a comprehensive analysis focusing on how it can enhance military capabilities, "
        "improve strategic advantages, and optimize operational effectiveness. "
        "Format the response as follows:\n\n"
        "Military Application 1: [Title]\n\n"
        "Enhancement of Capabilities: [Description]\n"
        "Strategic Advantage: [Description]\n"
        "Operational Optimization: [Description]\n\n"
        "Military Application 2: [Title]\n\n"
        "Enhancement of Capabilities: [Description]\n"
        "Strategic Advantage: [Description]\n"
        "Operational Optimization: [Description]\n\n"
        "Military Application 3: [Title]\n\n"
        "Enhancement of Capabilities: [Description]\n"
        "Strategic Advantage: [Description]\n"
        "Operational Optimization: [Description]"
    )
    prompt = f"{instruction_prompt}\n\nText:\n{text}"
    llm = Ollama(model="hf.co/Chidera123/model", request_timeout=500.0)
    response = llm.complete(prompt)
    impact_predictions = response.text.strip()
    return impact_predictions

def main():
    input_folder = "./fixed_summaries"
    output_folder = "./impact"
    
    if os.path.exists(output_folder):
        # Delete all files in the folder
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_folder)
    
    input_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    
    for filename in input_files:
        input_file_path = os.path.join(input_folder, filename)
        new_output_file_path = os.path.join(output_folder, f"impact_{filename}")
        
        raw_text = load_text_from_file(input_file_path)
        impact_predictions = predict_military_impact(raw_text)
        write_text_to_file(new_output_file_path, impact_predictions)
        print(f"Impact predictions have been saved to {new_output_file_path}")

if __name__ == "__main__":
    main()
