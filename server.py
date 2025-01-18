from flask import Flask, jsonify, request
import subprocess
from llama_index.llms.ollama import Ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/run-scripts', methods=['POST'])
def run_scripts():
    try:
        print("Running fetch_arxiv.py...")
        result1 = subprocess.run(['python', './Arxiv/fetch_arxiv.py'], capture_output=True, text=True)
        
        if result1.returncode == 0:
            print("fetch_arxiv.py completed successfully.")
            print("Running pdf_to_bart_summary.py...")
            result3 = subprocess.run(['python', './Arxiv/pdf_to_bart_summary.py'], capture_output=True, text=True)
            print("pdf_to_bart_summary.py completed.")
            
            if result3.returncode == 0:
                print("Running fix_summaries.py...")
                result4 = subprocess.run(['python', './Arxiv/fix_summaries.py'], capture_output=True, text=True)
                print("fix_summaries.py completed.")
                
                if result4.returncode == 0:
                    print("Running massw.py...")
                    result5 = subprocess.run(['python', './Arxiv/massw.py'], capture_output=True, text=True)
                    print("massw.py completed.")
                else:
                    print("fix_summaries.py failed.")
                    result5 = None
            else:
                print("pdf_to_bart_summary.py failed.")
                result4 = None
                result5 = None
            
            print("Running update_papers.py...")
            result2 = subprocess.run(['python', './Arxiv/update_papers.py'], capture_output=True, text=True)
            print("update_papers.py completed.")
            
            if result2.returncode != 0:
                print("update_papers.py failed.")
        else:
            print("fetch_arxiv.py failed.")
            result2 = None
            result3 = None
            result4 = None
            result5 = None
        
        return jsonify({
            'output1': result1.stdout,
            'error1': result1.stderr,
            'output2': result2.stdout if result2 else '',
            'error2': result2.stderr if result2 else 'fetch_arxiv.py failed, update_papers.py did not run.',
            'output3': result3.stdout if result3 else '',
            'error3': result3.stderr if result3 else 'update_papers.py failed, pdf_to_bart_summary.py did not run.',
            'output4': result4.stdout if result4 else '',
            'error4': result4.stderr if result4 else 'pdf_to_bart_summary.py failed, fix_summaries.py did not run.',
            'output5': result5.stdout if result5 else '',
            'error5': result5.stderr if result5 else 'fix_summaries.py failed, massw.py did not run.'
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    

class ChatbotMemory:
    def __init__(self):
        self.conversation_history = []

    def add_to_memory(self, user_input, assistant_response):
        self.conversation_history.append((user_input, assistant_response))

    def get_memory(self):
        return "\n".join([f"User: {user}\nAssistant: {assistant}" for user, assistant in self.conversation_history])

def query_rag(query: str):
    db = Chroma(
        persist_directory="./Chatbot/Recommendation Agent/chroma_db", 
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )
    results = db.similarity_search_with_score(query, k=3)
    context = "\n".join([result[0].page_content for result in results])
    return context

llm = Ollama(model="hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:latest", request_timeout=120.0)
memory = ChatbotMemory()

@app.route('/chat', methods=['POST'])
def chat_with_bot():
    user_question = request.json.get('message')
    if not user_question:
        return jsonify({'error': 'No message provided'}), 400

    # Get context from RAG (military knowledge)
    military_context = query_rag(user_question)
    
    # Load paper summaries (primary knowledge source)
    paper_summaries = ""
    summaries_dir = "./fixed_summaries"
    if os.path.exists(summaries_dir):
        for file in os.listdir(summaries_dir):
            if file.endswith(".txt"):
                with open(os.path.join(summaries_dir, file), 'r') as f:
                    paper_summaries += f.read() + "\n\n"
    
    memory_context = memory.get_memory()
    
    print(f"Military context: {military_context}")
    print(f"Memory context: {memory_context}")
    
    prompt = f"""
    You are a research assistant specializing in academic papers and military topics.
    Provide brief, focused answers using the available knowledge.
    
    Papers: {paper_summaries}
    Military Context: {military_context}
    History: {memory_context}
    
    User: {user_question}
    Assistant: Let me provide a concise answer based on the research papers and relevant military context.
    """
    
    try:
        response = llm.complete(prompt)
        print("LLM Response:", response)
        memory.add_to_memory(user_question, response)
        return jsonify({'response': response.text})
    except Exception as e:
        print("Error during LLM completion:", str(e))
        return jsonify({'error': 'Failed to generate response'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)