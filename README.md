# Healthpedia AI: Your Local Wellness Companion

<p align="center"> <img src="imgs/Medical Bot.png" alt="Healthpedia_AI" width="300" height="300"> </p>

## Short Description
Healthpedia AI is a locally-hosted medical chatbot that utilizes Retrieval-Augmented Generation (RAG) to provide accurate answers to health and medicine-related questions. Built with Flask, LangChain, Qdrant, and Ollama, it processes queries against a comprehensive medical knowledge base while keeping data private and local.

## Project Overview

Healthpedia AI is a comprehensive medical chatbot designed to answer a wide range of disease and medicine-related questions. It utilizes the extensive knowledge base provided by 'The GALE ENCYCLOPEDIA of MEDICINE' and employs advanced technologies to deliver accurate and insightful responses. 

<p align="center"> <img src="imgs/HealthpediaAI.png" alt="project_output" width="80%" height="80%"> </p>

## Tech Stack

- **Python:** Core programming language.
- **Flask:** Web application framework for the frontend interface.
- **LangChain:** Orchestrates the Retrieval-Augmented Generation (RAG) pipeline.
- **Ollama:** Runs the local LLM (e.g., Qwen2.5:3b) to generate and refine answers privately.
- **Qdrant:** Local Vector Database for storing and retrieving document embeddings.
- **HuggingFace Embeddings:** Generates semantic embeddings for chunks of medical texts.

## Project Flow

<p align="center"> <img src="imgs/medbot.drawio.png" alt="Project_Architecture" width="50%" height="50%"> </p>

1. **Load Corpus:** Import the contents of 'The GALE ENCYCLOPEDIA of MEDICINE'.
2. **Corpus Chunking:** Divide the medical corpus into manageable chunks.
3. **Embedding Conversion:** Use HuggingFace embeddings to convert chunks into vectors.
4. **Qdrant Indexing:** Store the vector embeddings locally in Qdrant for fast semantic search.
5. **Web Application (Flask):** The user interacts with the chatbot through a browser frontend.
6. **User Query Handling:** Retrieve relevant document chunks from the Qdrant vector database based on the user's question.
7. **LLM Integration (Ollama):** The user's question and the retrieved medical context are sent to the local LLM to generate a precise, context-aware answer.
8. **Display Output:** The refined answer is streamed back to the chatbot UI.

## Repository Structure

- `app.py`: Main Flask application handling the web server and LLM integration.
- `store_index.py`: Script to generate embeddings and populate the Qdrant database.
- `src/`: Contains helper scripts for downloading embeddings and prompt templates.
- `/data`: Directory to store the medical book corpus (`.pdf`).
- `/qdrant_data`: Local directory where Qdrant stores the vector database.
- `/templates` & `/static`: Frontend HTML, CSS, and JS files.

## Getting Started

1. Clone the repository: 
   ```bash
   git clone https://github.com/sameeerjadhav/HealthPedia-AI.git
   cd HealthPedia-AI
   ```
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
4. Install and start [Ollama](https://ollama.com/), then pull the required model (e.g., `qwen2.5:3b`):
   ```bash
   ollama pull qwen2.5:3b
   ```
5. Ingest the data (assuming you have your data in the `/data` folder):
   ```bash
   python store_index.py
   ```
6. Run the application: 
   ```bash
   python app.py
   ```
7. Open your browser and navigate to `http://127.0.0.1:5000/`.

## License

This project is licensed under the [MIT License](LICENSE).
