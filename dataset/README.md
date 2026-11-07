# PDF Question Answering System

## Project Overview
This project builds a PDF Question Answering System using LangChain, ChromaDB, Sentence Transformers, and Ollama.

## Objectives
- Extract text from PDF documents.
- Split text into smaller chunks.
- Generate vector embeddings.
- Store embeddings in ChromaDB.
- Retrieve relevant information.
- Generate answers using a local LLM through Ollama.

## Technologies Used
- Python
- LangChain
- ChromaDB
- Ollama
- Sentence Transformers
- PyPDF

## Project Structure
- dataset – Stores PDF files and CSV outputs.
- chroma_db – Stores vector embeddings.
- notebook – Contains the Jupyter Notebook.
- pdf_qa.py – Main Python program.
- requirements.txt – Required libraries.

## Expected Output
The system accepts questions about a PDF and generates answers based on the document content.