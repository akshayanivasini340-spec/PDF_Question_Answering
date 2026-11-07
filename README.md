# PDF Question Answering System using LangChain, ChromaDB and Ollama

## Project Overview
This project is a PDF Question Answering System that allows users to ask questions based on the content of a PDF document. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate accurate answers using a local Large Language Model (LLM).

## Objectives
- Extract text from PDF documents.
- Split text into smaller chunks.
- Generate embeddings using Sentence Transformers.
- Store embeddings in ChromaDB.
- Retrieve the top 5 relevant chunks for a question.
- Generate answers using Ollama.
- Compare answers with and without retrieval.
- Save questions and answers in CSV files.

## Technologies Used
- Python
- LangChain
- ChromaDB
- Ollama
- Sentence Transformers
- Pandas
- Jupyter Notebook

## Project Structure
```text
PDF_Question_Answering/
│
├── dataset/
│   ├── sample.pdf
│   ├── extracted_text.txt
│   ├── chunks.csv
│   ├── queries.csv
│   ├── answers.csv
│   └── notebook_answers.csv
│
├── notebook/
│   └── Week13_PDF_Question_Answering.ipynb
│
├── pdf_qa.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Open the project folder:
```bash
cd PDF_Question_Answering
```

3. Install the required libraries:
```bash
pip install -r requirements.txt
```

4. Install and run Ollama.

5. Download the Llama 3.2 model:
```bash
ollama pull llama3.2
```

## Execution

Run the Python program:
```bash
python pdf_qa.py
```

To execute the Jupyter Notebook, open:
`notebook/Week13_PDF_Question_Answering.ipynb`

## Working Process

1. PDF text extraction
2. Text splitting into chunks
3. Embedding generation
4. Vector storage using ChromaDB
5. Retrieval of top 5 relevant chunks
6. Answer generation using Ollama
7. Comparison of answers with and without retrieval
8. Saving results in CSV files

## Output
- Extracted PDF text
- 709 text chunks
- Generated embeddings
- ChromaDB vector database
- Top 5 retrieved chunks
- Answers with and without retrieval
- CSV files containing questions and answers

## Conclusion
The project demonstrates how Retrieval-Augmented Generation (RAG) can be used to build a PDF Question Answering System. It retrieves relevant information from PDF documents and generates context-aware answers using a locally running LLM.

## Author
Student Project – Week 13: PDF Question Answering System
