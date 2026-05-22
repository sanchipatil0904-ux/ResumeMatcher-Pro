# ============================================
# AI Resume Shortlisting System
# Run in VS Code using Streamlit
# ============================================

# Install Required Libraries:
# pip install streamlit PyPDF2 scikit-learn pandas

import streamlit as st
import PyPDF2
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================
# Function to Extract Text from PDF
# ============================================

def extract_text_from_pdf(uploaded_file):

    text = ""

    try:
        reader = PyPDF2.PdfReader(uploaded_file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    except Exception as e:
        st.error(f"Error Reading PDF: {e}")

    return text


# ============================================
# Streamlit UI
# ============================================

st.set_page_config(
    page_title="AI Resume Shortlisting System",
    layout="wide"
)

st.title("📄 AI-Based Resume Shortlisting System")

st.write("Upload resumes and compare them with Job Description using NLP.")


# ============================================
# Job Description Input
# ============================================

job_description = st.text_area(
    "Enter Job Description",
    height=200
)


# ============================================
# Upload Multiple Resume PDFs
# ============================================

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================
# Analyze Button
# ============================================

if st.button("Analyze Resumes"):

    if not job_description:
        st.warning("Please enter Job Description")

    elif not uploaded_files:
        st.warning("Please upload at least one resume")

    else:

        results = []

        for uploaded_file in uploaded_files:

            resume_text = extract_text_from_pdf(uploaded_file)

            # TF-IDF Similarity
            documents = [job_description, resume_text]

            tfidf = TfidfVectorizer()

            tfidf_matrix = tfidf.fit_transform(documents)

            similarity = cosine_similarity(
                tfidf_matrix[0:1],
                tfidf_matrix[1:2]
            )

            score = round(similarity[0][0] * 100, 2)

            results.append({
                "Resume": uploaded_file.name,
                "Match Score (%)": score
            })


        # ============================================
        # Ranking Results
        # ============================================

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="Match Score (%)",
            ascending=False
        )

        st.success("Analysis Completed ✅")

        st.subheader("📊 Resume Ranking")

        st.dataframe(results_df, use_container_width=True)


        # ============================================
        # Best Candidate
        # ============================================

        top_candidate = results_df.iloc[0]

        st.subheader("🏆 Best Matched Candidate")

        st.write(
            f"**{top_candidate['Resume']}** "
            f"with score "
            f"**{top_candidate['Match Score (%)']}%**"
        )
