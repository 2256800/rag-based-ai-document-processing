# RAG-Based AI Document Processing Pipeline

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to process financial PDF documents and organize their content for retrieval and analysis.

The pipeline uses Python to extract and process document content, supports PDF text and table extraction, and uses embeddings and vector storage to support relevant information retrieval.

## Key Features

* PDF document processing
* Text and table extraction
* Document and visual-content processing
* Hugging Face embeddings
* Chroma vector database
* Retrieval-Augmented Generation (RAG)
* Modular Python implementation

## Technologies Used

* Python
* LangChain
* Hugging Face
* ChromaDB
* PyMuPDF
* pdfplumber
* Pillow

## Project Structure

```text
ragproject_pipeline/
├── rag.py
├── detection.py
├── retrieval.py
├── requirements.txt
├── README.md
├── .gitignore
├── output_images/
├── vector_db/
└── financial_data.pdf
```

Note: The `venv/` directory is excluded from version control. The `vector_db/` directory contains generated data and can be recreated when necessary.

## Installation

1. Clone this repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Navigate to the project directory:

   ```bash
   cd ragproject_pipeline
   ```

3. Create and activate a virtual environment.

   Windows:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How It Works

1. Load the financial PDF document.
2. Extract relevant document content.
3. Process the extracted information.
4. Generate embeddings and store searchable representations in the vector database.
5. Retrieve relevant content when a query is submitted.
6. Use the retrieved information in the RAG workflow.

## Learning Outcomes

* Understanding document-processing pipelines
* Working with embeddings and vector databases
* Learning retrieval-based AI architectures
* Organizing Python applications into reusable modules

## Author

B.Tech Artificial Intelligence and Data Science Graduate
