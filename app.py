import streamlit as st
import pandas as pd
import plotly.express as px
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gtts import gTTS
from googletrans import Translator
import speech_recognition as sr
import tempfile
import os

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    nco_df = pd.read_csv("nco_cleaned.csv")
    govt_df = pd.read_csv("govt_schemes.csv")
    return nco_df, govt_df

nco_df, govt_df = load_data()

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ------------------ Build FAISS Index ------------------
@st.cache_resource
def build_index(nco_df):
    embeddings = model.encode(nco_df['title'].tolist())
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

index = build_index(nco_df)

# ------------------ Search Function ------------------
def semantic_search(query, top_k=5):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    return nco_df.iloc[indices[0]]

# ------------------ Multilingual Translation ------------------
def translate_text(text, target_lang):
    translator = Translator()
    return translator.translate(text, dest=target_lang).text

# ------------------ Voice Input ------------------
def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        st.error("Sorry, could not understand audio.")
        return ""

# ------------------ Tabs Layout ------------------
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Charts", "💬 Chatbot"])

# ------------------ Tab 1: Search ------------------
with tab1:
    st.title("🔍 NCO Job Search + Govt Scheme Recommender")
    
    col1, col2 = st.columns([3,1])
    with col1:
        search_option = st.radio("Search type:", ["Text", "Voice"])
        if search_option == "Text":
            query = st.text_input("Enter job title or skill:")
        else:
            if st.button("🎤 Record Voice"):
                query = voice_to_text()
            else:
                query = ""
    
    with col2:
        top_k = st.slider("Number of results", 1, 10, 5)
    
    if st.button("Search"):
        if query:
            results = semantic_search(query, top_k)
            st.subheader("Matching Jobs")
            st.dataframe(results)

            # Recommend govt schemes
            st.subheader("Recommended Government Schemes")
            matched_schemes = govt_df[govt_df['eligible_nco_code'].isin(results['nco_code'])]
            st.dataframe(matched_schemes)

            # Text-to-Speech output
            if st.checkbox("🔊 Read results aloud"):
                tts = gTTS(f"Found {len(results)} matching jobs for your query.")
                tts.save("output.mp3")
                st.audio("output.mp3")
        else:
            st.warning("Please enter a search query.")

# ------------------ Tab 2: Charts ------------------
with tab2:
    st.title("📊 Job & Scheme Analytics")
    job_count = nco_df['title'].value_counts().head(10)
    fig = px.bar(job_count, x=job_count.index, y=job_count.values, title="Top Job Titles")
    st.plotly_chart(fig)

# ------------------ Tab 3: Chatbot ------------------
with tab3:
    st.title("💬 Multilingual Career Chatbot")
    user_input = st.text_input("Ask me anything about NCO jobs or govt schemes:")
    lang = st.selectbox("Translate my reply to:", ["en", "ta", "hi", "ml", "te"])
    
    if st.button("Get Reply"):
        if user_input:
            # Here you can integrate OpenAI/GPT API if needed
            reply = f"This is a placeholder reply for: {user_input}"
            translated = translate_text(reply, lang)
            st.write(translated)
        else:
            st.warning("Please enter a question.")
