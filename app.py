pip install streamlit pandas plotly faiss-cpu sentence-transformers numpy scikit-learn gTTS googletrans==4.0.0-rc1 SpeechRecognition
import streamlit as st
import pandas as pd
import plotly.express as px
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from googletrans import Translator
from gtts import gTTS
import speech_recognition as sr
import os

# =========================
# CACHE MODEL LOADING
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
translator = Translator()

# =========================
# LOAD DATASETS
# =========================
@st.cache_data
def load_data():
    nco_df = pd.read_csv("nco_cleaned.csv")
    schemes_df = pd.read_csv("govt_schemes.csv")
    return nco_df, schemes_df

nco_df, schemes_df = load_data()

# =========================
# CREATE EMBEDDINGS & INDEX
# =========================
@st.cache_resource
def create_index(data):
    embeddings = model.encode(data['title'].tolist(), convert_to_tensor=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

index, embeddings = create_index(nco_df)

# =========================
# SEARCH FUNCTION
# =========================
def search_jobs(query, top_k=5, lang='en'):
    if lang != 'en':
        query = translator.translate(query, src=lang, dest='en').text
    query_vector = model.encode([query], convert_to_tensor=False)
    D, I = index.search(np.array(query_vector).astype('float32'), top_k)
    results = nco_df.iloc[I[0]]
    return results

# =========================
# SCHEME MATCHING
# =========================
def match_schemes(nco_codes):
    return schemes_df[schemes_df['eligible_nco_code'].isin(nco_codes)]

# =========================
# VOICE INPUT VIA FILE UPLOAD
# =========================
def voice_input_from_file(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        st.success(f"Recognized Speech: {text}")
        return text
    except:
        st.error("Could not process the audio file.")
        return ""

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="NCO Job Search & Govt Scheme Recommender", layout="wide")

tab1, tab2, tab3 = st.tabs(["🔍 Job & Scheme Search", "📊 Charts & Insights", "🤖 Multilingual Voice Chatbot"])

# ========== TAB 1: Search ==========
with tab1:
    st.header("🔍 NCO Job Search + Govt Scheme Recommender")
    query = st.text_input("Enter job title/skill:")
    num_results = st.slider("Number of results", 1, 10, 5)
    lang_choice = st.selectbox("Select Language", ["en", "hi", "ta", "te", "ml"])

    uploaded_audio = st.file_uploader("Upload voice query (WAV/MP3)", type=["wav", "mp3"])
    if uploaded_audio:
        query = voice_input_from_file(uploaded_audio)

    if st.button("Search"):
        if query:
            jobs = search_jobs(query, num_results, lang_choice)
            schemes = match_schemes(jobs['nco_code'].tolist())
            st.subheader("Matching Jobs")
            st.dataframe(jobs)
            st.subheader("Eligible Schemes")
            st.dataframe(schemes)
        else:
            st.warning("Please enter a query or upload voice.")

# ========== TAB 2: Charts ==========
with tab2:
    st.header("📊 Data Insights")
    fig = px.histogram(nco_df, x="title", title="Job Title Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ========== TAB 3: Chatbot ==========
with tab3:
    st.header("🤖 Multilingual Voice Chatbot")
    user_input = st.text_input("Type your question:")
    audio_upload_chat = st.file_uploader("Upload voice message (WAV/MP3)", type=["wav", "mp3"])
    if audio_upload_chat:
        user_input = voice_input_from_file(audio_upload_chat)
    
    if user_input:
        translated_text = translator.translate(user_input, dest="en").text
        bot_reply = f"I found {len(search_jobs(translated_text))} matching jobs for your query."
        st.write("💬 Bot:", bot_reply)

        # Text-to-Speech
        tts = gTTS(text=bot_reply, lang='en')
        tts.save("bot_reply.mp3")
        st.audio("bot_reply.mp3", format="audio/mp3")
