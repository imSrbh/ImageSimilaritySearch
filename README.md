# Image Similarity Search App

This project demonstrates an image similarity search application using the CLIP model by OpenAI.

## Overview

The project consists of the following components:

- `app.py`: Streamlit web application for image search.
- `get_embeddings.py`: Script to preprocess images and extract CLIP embeddings.
- `data/`: Directory possibly containing additional data used in the project.
- `requirements.txt`: File listing Python dependencies.

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:imSrbh/ImageSimilaritySearch.git
   cd ImageSimilaritySearch
   python3 -m venv env
   source env/bin/activate
   pip3 install -r requirements.txt
   ```
   > Note: replace the folder_path of images

2. Generate Embeddings:
   ```bash
   python3 get_embedding.py

3. Run streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## TODO and Future Roadmap:

- [ ] Add Vector databases Support(for eg. Postgres, SingleStore, ChromaDB, Pinecone etc...)
- [ ] Add provision in the streamlit app to switch between datasets to search on.
- [ ] Benchmark the Vector databases
- [ ] Write a Blog post
- [ ] Add github action workflow 
