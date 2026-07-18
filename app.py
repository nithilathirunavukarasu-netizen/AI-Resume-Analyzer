
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
STOPWORDS = ["the", "is", "and", "to", "in", "of", "for", "with", "on", "at", "by", "an", "be", "this", "that"]
SKILLS = ["python", "java", "sql", "machine learning", "ai", "data science", "aws", "cloud", "javascript"]

# --- Functions ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return " ".join([w for w in text.split() if w not in STOPWORDS])

# --- Streamlit UI ---
st.title("📄 AI Resume Matcher")

# Fixed the file uploader syntax
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if resume_file and job_desc:
        # 1. Extract & Clean (Now runs only after file is uploaded)
        raw_text = extract_text_from_pdf(resume_file)
        resume_clean = clean_text(raw_text)
        jd_clean = clean_text(job_desc)

        # 2. Similarity Math
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, jd_clean])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]

        # 3. Results
        st.success(f"Match Score: {round(score * 100, 2)}%")

        # Skill Matching
        found = [s for s in SKILLS if s in resume_clean]
        st.write("**Detected Skills:**", ", ".join(found) if found else "None")
    else:
        st.error("Please upload a PDF and paste a Job Description first!")