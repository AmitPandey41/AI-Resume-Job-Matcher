
import os
import re
import pickle
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume–Job Matcher",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_ml_model():

    with open(
        "models/tfidf_vectorizer.pkl",
        "rb"
    ) as f:
        vectorizer = pickle.load(f)

    with open(
        "models/resume_classifier.pkl",
        "rb"
    ) as f:
        model = pickle.load(f)

    return vectorizer, model


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_sentence_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# SEMANTIC MATCH
# ============================================================

def calculate_semantic_match(
    resume_text,
    job_description
):

    model = load_sentence_model()

    resume_embedding = model.encode(
        resume_text,
        convert_to_numpy=True
    )

    job_embedding = model.encode(
        job_description,
        convert_to_numpy=True
    )

    score = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    score = max(0, min(1, score))

    return round(score * 100, 2)


# ============================================================
# SKILL DATABASE
# ============================================================

SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "sql",
    "mysql",
    "mongodb",
    "pandas",
    "numpy",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "generative ai",
    "natural language processing",
    "computer vision",
    "transformers",
    "llm",
    "large language models",
    "prompt engineering",
    "tensorflow",
    "pytorch",
    "cnn",
    "lstm",
    "rnn",
    "git",
    "github",
    "streamlit",
    "rest apis",
    "html",
    "css",
    "react",
    "node.js",
    "flask",
    "docker",
    "tableau",
    "data analysis",
    "data science",
    "anomaly detection",
    "feature engineering",
    "classification",
    "regression",
    "clustering"
]


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):

    text = text.lower()

    detected = []

    for skill in SKILLS:

        if skill.lower() in text:

            detected.append(skill)

    return sorted(set(detected))


# ============================================================
# SKILL MATCH
# ============================================================

def calculate_skill_match(
    resume_text,
    job_description
):

    resume_skills = extract_skills(
        resume_text
    )

    job_skills = extract_skills(
        job_description
    )

    matched = [
        skill
        for skill in job_skills
        if skill in resume_skills
    ]

    missing = [
        skill
        for skill in job_skills
        if skill not in resume_skills
    ]

    if len(job_skills) == 0:

        skill_score = 0.0

    else:

        skill_score = (
            len(matched) /
            len(job_skills)
        ) * 100

    return (
        matched,
        missing,
        round(skill_score, 2)
    )


# ============================================================
# GEMINI FEEDBACK
# ============================================================

def generate_gemini_feedback(
    resume_text,
    job_description,
    overall_score,
    match_level,
    matched_skills,
    missing_skills,
    predicted_category
):

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        return (
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY in Streamlit secrets."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
You are an expert AI/ML career and resume analyst.

Analyze the candidate's resume against the job description.

Predicted Resume Category:
{predicted_category}

Overall Match:
{overall_score}%

Match Level:
{match_level}

Matched Skills:
{", ".join(matched_skills)}

Missing Skills:
{", ".join(missing_skills)}

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Give professional and realistic feedback.

Use exactly these sections:

1. Overall Assessment
2. Resume Strengths
3. Missing or Weak Skills
4. Resume Improvement Suggestions
5. Interview Preparation Suggestions

Do not invent experience or skills that are not present.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# ============================================================
# HEADER
# ============================================================

st.title("🤖 AI Resume–Job Matcher")

st.write(
    "Analyze your resume against a job description "
    "using Machine Learning, Deep Learning and Generative AI."
)


# ============================================================
# INPUTS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader("📄 Upload Resume")

    uploaded_file = st.file_uploader(
        "Upload your resume PDF",
        type=["pdf"]
    )


with col2:

    st.subheader("📝 Job Description")

    job_description = st.text_area(
        "Paste the complete job description",
        height=250
    )


# ============================================================
# ANALYZE
# ============================================================

if st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True
):

    if uploaded_file is None:

        st.error(
            "Please upload your resume PDF."
        )

        st.stop()


    if not job_description.strip():

        st.error(
            "Please enter the job description."
        )

        st.stop()


    with st.spinner(
        "Analyzing your resume..."
    ):

        # Extract PDF
        resume_text = extract_resume_text(
            uploaded_file
        )

        if not resume_text.strip():

            st.error(
                "Could not extract text from the PDF."
            )

            st.stop()


        # ML prediction
        vectorizer, ml_model = load_ml_model()

        cleaned_resume = clean_text(
            resume_text
        )

        resume_vector = vectorizer.transform(
            [cleaned_resume]
        )

        predicted_category = ml_model.predict(
            resume_vector
        )[0]


        # Semantic score
        semantic_score = calculate_semantic_match(
            resume_text,
            job_description
        )


        # Skill score
        (
            matched_skills,
            missing_skills,
            skill_score
        ) = calculate_skill_match(
            resume_text,
            job_description
        )


        # Overall score
        overall_score = round(
            (semantic_score * 0.50)
            +
            (skill_score * 0.50),
            2
        )


        # Match level
        if overall_score >= 85:

            match_level = "Excellent Match"

        elif overall_score >= 70:

            match_level = "Good Match"

        elif overall_score >= 50:

            match_level = "Moderate Match"

        else:

            match_level = "Low Match"


        # Gemini
        ai_feedback = generate_gemini_feedback(
            resume_text=resume_text,
            job_description=job_description,
            overall_score=overall_score,
            match_level=match_level,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            predicted_category=predicted_category
        )


    # ========================================================
    # RESULTS
    # ========================================================

    st.success(
        "✅ Resume analysis completed!"
    )


    st.divider()

    st.subheader("📊 Analysis Results")


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Predicted Category",
            predicted_category
        )


    with c2:

        st.metric(
            "Semantic Match",
            f"{semantic_score}%"
        )


    with c3:

        st.metric(
            "Skill Match",
            f"{skill_score}%"
        )


    with c4:

        st.metric(
            "Overall Match",
            f"{overall_score}%"
        )


    st.subheader(
        f"🎯 Match Level: {match_level}"
    )


    # ========================================================
    # SKILLS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("✅ Matched Skills")

        if matched_skills:

            for skill in matched_skills:

                st.write(
                    f"✓ {skill}"
                )

        else:

            st.write(
                "No matched skills found."
            )


    with col2:

        st.subheader("⚠️ Missing Skills")

        if missing_skills:

            for skill in missing_skills:

                st.write(
                    f"• {skill}"
                )

        else:

            st.write(
                "No major missing skills found."
            )


    # ========================================================
    # GEMINI FEEDBACK
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 Generative AI Feedback"
    )

    st.markdown(
        ai_feedback
    )


    # ========================================================
    # RESUME PREVIEW
    # ========================================================

    with st.expander(
        "📄 View Extracted Resume Text"
    ):

        st.text(
            resume_text
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Resume–Job Matcher | "
    "Machine Learning • Deep Learning • Generative AI"
)
