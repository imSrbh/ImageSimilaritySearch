import pickle
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load embeddings from file
with open('embeddings_train.pkl', 'rb') as f:
    image_embeddings = pickle.load(f)

# Ensure each embedding is converted back to numpy array
image_embeddings = {path: torch.tensor(emb) if isinstance(emb, list) else emb for path, emb in image_embeddings.items()}

# Streamlit app configuration
st.header("Image Search App")
search_term = "a picture of " + st.text_input("Search: ")
search_embedding = model.get_text_features(**preprocess(text=search_term, return_tensors="pt")).cpu().detach().numpy()

st.sidebar.header("App Settings")
top_number = st.sidebar.slider("Number of Search Results", min_value=1, max_value=30)
picture_width = st.sidebar.slider("Picture Width", min_value=100, max_value=500)

# Rank images by similarity
df_rank = pd.DataFrame(columns=["image_path", "sim_score"])

for path, embedding in image_embeddings.items():
    # Ensure embedding is numpy array
    if isinstance(embedding, list):
        embedding = torch.tensor(embedding).numpy()
    sim = cosine_similarity(embedding.reshape(1, -1), search_embedding.reshape(1, -1)).flatten().item()
    df_rank = pd.concat([df_rank, pd.DataFrame(data=[[path, sim]], columns=["image_path", "sim_score"])])

df_rank.reset_index(inplace=True, drop=True)
df_rank.sort_values(by="sim_score", ascending=False, inplace=True, ignore_index=True)

# Display search results
col1, col2, col3 = st.columns(3)
df_result = df_rank.head(top_number)

for i in range(top_number):
    if i % 3 == 0:
        with col1:
            st.image(df_result.loc[i, "image_path"], width=picture_width)
    elif i % 3 == 1:
        with col2:
            st.image(df_result.loc[i, "image_path"], width=picture_width)
    elif i % 3 == 2:
        with col3:
            st.image(df_result.loc[i, "image_path"], width=picture_width)
