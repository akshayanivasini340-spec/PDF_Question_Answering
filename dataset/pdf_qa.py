
import os
import requests
import pandas as pd
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# PROJECT PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(BASE_DIR, "dataset", "sample.pdf")
TEXT_PATH = os.path.join(BASE_DIR, "dataset", "extracted_text.txt")
CHUNKS_PATH = os.path.join(BASE_DIR, "dataset", "chunks.csv")
ANSWERS_PATH = os.path.join(BASE_DIR, "dataset", "answers.csv")
QUERIES_PATH = os.path.join(BASE_DIR, "dataset", "queries.csv")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# 1. EXTRACT TEXT FROM PDF

def extract_pdf_text():

    print("\n1. EXTRACTING TEXT FROM PDF...")

    if not os.path.exists(PDF_PATH):
        print("ERROR: sample.pdf not found!")
        return None

    reader = PdfReader(PDF_PATH)
    extracted_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_text += page_text + "\n"

    with open(TEXT_PATH, "w", encoding="utf-8") as file:
        file.write(extracted_text)

    print("PDF extraction completed!")
    print("Total pages:", len(reader.pages))
    print("Total characters:", len(extracted_text))

    return extracted_text


# 2. SPLIT TEXT INTO CHUNKS

def split_text_into_chunks(text):

    print("\n2. SPLITTING TEXT INTO CHUNKS...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    df = pd.DataFrame({
        "chunk_id": range(len(chunks)),
        "text": chunks
    })

    df.to_csv(CHUNKS_PATH, index=False)

    print("Text chunking completed!")
    print("Total chunks:", len(chunks))

    return chunks


# 3. GENERATE EMBEDDINGS

def generate_embeddings(chunks):

    print("\n3. GENERATING EMBEDDINGS...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("Embeddings generated successfully!")

    return model, embeddings


# 4. STORE DATA IN CHROMADB

def store_in_chromadb(chunks, embeddings):

    print("\n4. STORING DATA IN CHROMADB...")

    os.makedirs(CHROMA_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection("pdf_documents")
    except Exception:
        pass

    collection = client.create_collection(
        name="pdf_documents"
    )

    batch_size = 500

    for start in range(0, len(chunks), batch_size):

        end = min(start + batch_size, len(chunks))

        collection.add(
            ids=[str(i) for i in range(start, end)],
            documents=chunks[start:end],
            embeddings=embeddings[start:end].tolist()
        )

        print("Stored chunks:", end, "/", len(chunks))

    print("Data stored successfully in ChromaDB!")

    return collection


# 5. RETRIEVE TOP 5 CHUNKS

def retrieve_chunks(question, collection, embedding_model):

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(5, collection.count())
    )

    retrieved_chunks = results["documents"][0]

    print("\nTop", len(retrieved_chunks), "chunks retrieved!")

    return retrieved_chunks


# 6. GENERATE ANSWER USING OLLAMA

def generate_answer(prompt):

    url = "http://localhost:11434/api/generate"

    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(
            url,
            json=data,
            timeout=300
        )

        response.raise_for_status()

        return response.json()["response"]

    except requests.exceptions.RequestException as error:

        print("Ollama connection error:", error)

        return "Unable to connect to Ollama. Check whether Ollama is running."


# 7. ANSWER WITHOUT RETRIEVAL

def answer_without_retrieval(question):

    prompt = f"""
Answer the question using your general knowledge.

Question:
{question}

Answer:
"""

    return generate_answer(prompt)


# 8. ANSWER WITH RETRIEVAL

def answer_with_retrieval(question, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer the question using ONLY the information in the context.

If the answer is not available in the context,
say: "The answer is not available in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    return generate_answer(prompt)


# 9. SAVE RESULTS IN CSV

def save_results(question, chunks, rag_answer, normal_answer):

    result = {
        "Question": question,
        "Retrieved_Chunks": "\n\n".join(chunks),
        "Answer_With_Retrieval": rag_answer,
        "Answer_Without_Retrieval": normal_answer
    }

    df = pd.DataFrame([result])

    if os.path.exists(ANSWERS_PATH):

        df.to_csv(
            ANSWERS_PATH,
            mode="a",
            header=False,
            index=False
        )

    else:

        df.to_csv(
            ANSWERS_PATH,
            index=False
        )

    print("\nResults saved in answers.csv!")


# 10. PROCESS QUESTION

def process_question(question, collection, embedding_model):

    print("\n" + "=" * 50)
    print("QUESTION:", question)
    print("=" * 50)

    chunks = retrieve_chunks(
        question,
        collection,
        embedding_model
    )

    print("\nGenerating answer without retrieval...")

    normal_answer = answer_without_retrieval(question)

    print("\nGenerating answer with retrieval...")

    rag_answer = answer_with_retrieval(
        question,
        chunks
    )

    print("\nANSWER WITHOUT RETRIEVAL:")
    print(normal_answer)

    print("\nANSWER WITH RETRIEVAL:")
    print(rag_answer)

    save_results(
        question,
        chunks,
        rag_answer,
        normal_answer
    )


# 11. MAIN PROGRAM

def main():

    print("\n====================================")
    print(" PDF QUESTION ANSWERING SYSTEM")
    print("====================================")

    text = extract_pdf_text()

    if not text or not text.strip():
        print("No text found in PDF!")
        return

    chunks = split_text_into_chunks(text)

    if not chunks:
        print("No chunks found!")
        return

    embedding_model, embeddings = generate_embeddings(chunks)

    collection = store_in_chromadb(chunks, embeddings)

    print("\nPDF QUESTION ANSWERING SYSTEM IS READY!")

    # Process questions from queries.csv

    if os.path.exists(QUERIES_PATH):

        questions_df = pd.read_csv(QUERIES_PATH)

        if "question" in questions_df.columns:

            for question in questions_df["question"].dropna():

                if str(question).strip():

                    process_question(
                        str(question),
                        collection,
                        embedding_model
                    )

    # Interactive question answering

    print("\nAsk questions about your PDF.")
    print("Type exit to stop.")

    while True:

        question = input("\nEnter your question: ").strip()

        if question.lower() == "exit":

            print("Thank you for using PDF Question Answering!")
            break

        if question:

            process_question(
                question,
                collection,
                embedding_model
            )


# START PROGRAM

if __name__ == "__main__":
    main()
