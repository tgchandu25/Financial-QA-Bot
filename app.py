import gradio as gr
import pandas as pd
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Function to process and embed extracted text
def create_faiss_index(text_data):
    sentences = text_data.split("\n")
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, sentences

# Function to retrieve the most relevant financial data
def get_financial_answer(query, index, sentences):
    query_embedding = model.encode([query])
    _, top_indices = index.search(np.array(query_embedding), 1)
    return sentences[top_indices[0][0]]

# Global variables
faiss_index = None
sentence_data = None

# Upload and process PDF function
def upload_pdf(file):
    global faiss_index, sentence_data
    text_data = extract_text_from_pdf(file.name)
    faiss_index, sentence_data = create_faiss_index(text_data)
    return "File processed successfully. You can now ask financial questions."

# Function to answer user queries
def answer_query(question):
    if faiss_index is None:
        return "Please upload a financial document first."
    return get_financial_answer(question, faiss_index, sentence_data)

# Gradio UI
with gr.Blocks() as financial_qa_bot:
    gr.Markdown("# 📊 Financial QA Bot 🤖")
    gr.Markdown("Ask financial questions based on uploaded financial documents.")

    with gr.Row():
        file_upload = gr.File(label="Upload Financial PDF")
        upload_button = gr.Button("Extract Data")

    upload_status = gr.Textbox(label="Extraction Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(label="Query", placeholder="Ask a financial question...")
        answer_output = gr.Textbox(label="Output", interactive=False)

    submit_button = gr.Button("Submit")

    upload_button.click(upload_pdf, inputs=file_upload, outputs=upload_status)
    submit_button.click(answer_query, inputs=question_input, outputs=answer_output)

financial_qa_bot.launch()
