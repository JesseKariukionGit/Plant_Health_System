# 🌿 Plant Health Diagnosis and Recommendation System

An end-to-end AI system that diagnoses plant leaf diseases from a photo and generates a tailored, three-part treatment plan using a CNN classifier combined with a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- **Image Upload** — Upload a photo of a plant leaf (JPG/PNG)
- **AI Disease Classification** — An EfficientNet-B0 CNN trained on 38 disease classes identifies the issue with ~88% accuracy
- **AI-Generated Treatment Plans** — A RAG pipeline retrieves relevant context from 76 agricultural reference PDFs and uses a Groq-hosted Llama 3 model to generate a structured plan covering:
  - Organic / Cultural treatment
  - Chemical / Fungicide treatment
  - Prevention / Sanitation
- **Diagnosis History** — Past diagnoses are stored and viewable with pagination (7 per page), automatically expiring after 24 hours

## Tech Stack

| Layer | Technologies |
|---|---|
| Backend | Python 3.9, FastAPI, Uvicorn |
| Machine Learning | TensorFlow / Keras (EfficientNet-B0) |
| RAG Pipeline | LangChain, ChromaDB, HuggingFace Embeddings (`all-MiniLM-L6-v2`), Groq API (Llama 3) |
| Frontend | Vanilla HTML, CSS, JavaScript (served as static files by FastAPI) |
| Database | SQLite (diagnosis history) |
| Config | `python-dotenv` for secrets management |

## How It Works

1. A user uploads a photo of a plant leaf through the web interface.
2. The EfficientNet-B0 model classifies the leaf into one of 38 disease categories.
3. The diagnosed condition is used to query a vector store (ChromaDB) built from 76 agricultural PDFs, retrieving the most relevant reference material.
4. The retrieved context is passed to a Llama 3 model (via Groq) to generate a structured, three-part treatment recommendation.
5. The diagnosis and recommendation are saved to a local SQLite database and shown in the history view.

## Getting Started

### Prerequisites

- Python 3.9
- A [Groq API key](https://console.groq.com)

### Installation

```bash
git clone https://github.com/JesseKariukionGit/Plant_Health_System.git
cd Plant_Health_System

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Key dependencies: `fastapi`, `uvicorn`, `tensorflow`, `langchain`, `langchain-groq`, `chromadb`, `pypdf`, `python-dotenv`

### Model Files

Make sure the following are present in `models/`:
- `efficientnetb0_plantvillage.h5`
- `class_names.json`

> Note: `.h5` model files are tracked via [Git LFS](https://git-lfs.github.com) in this repo. Make sure Git LFS is installed before cloning: `git lfs install`

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### Running the App

```bash
cd backend
python3 app.py
```

The backend starts at `http://localhost:8000`.

Open your browser to:

```
http://localhost:8000/static/index.html
```

Upload a plant leaf image and click **Analyze Plant** to receive a diagnosis and treatment plan.

## Project Purpose

This project was built as a full-stack demonstration of applied AI — combining a computer vision classifier, a RAG pipeline, and an LLM into a single, practical tool for plant disease diagnosis. It showcases:

- End-to-end AI integration (CNN + RAG + LLM)
- Full-stack web development (FastAPI + vanilla frontend)
- Use of modern AI tooling (LangChain, ChromaDB, Groq)
- A real-world, applicable problem space

## License

Add your preferred license here (e.g. MIT).
