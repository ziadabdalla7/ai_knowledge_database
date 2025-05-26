import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess
from utils.retrieval import create_embeddings, build_faiss_index, search

# Load data
df = pd.read_csv("data/otrs_tickets.csv")
df['cleaned'] = df['description'].apply(preprocess)

# Build embeddings and FAISS index
embeddings = create_embeddings(df['cleaned'].tolist())
index = build_faiss_index(embeddings)

st.title("AI Knowledge Base for Support Tickets")
query = st.text_input("Enter a support query:")

if query:
    results = search(index, query, df['description'].tolist())
    st.subheader("Top Results")
    for i, (text, score) in enumerate(results):
        st.markdown(f"**Result {i+1}:**")
        st.text(text)
        st.caption(f"Similarity score: {round(score, 4)}")
