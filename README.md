🤖 AI Resume–Job Matcher
An AI-powered web app that analyzes how well a resume matches a job description — combining Machine Learning, Deep Learning (embeddings), and Generative AI feedback.
🔗 Live Demo: ai-resume-job-matcher-aafmhafnnd346armjtk4nn.streamlit.app
Features
📄 Resume category prediction — TF-IDF + Logistic Regression / Random Forest classifier trained on labeled resume data
🧠 Semantic match scoring — Sentence-Transformers (`all-MiniLM-L6-v2`) embeddings + cosine similarity between resume and job description
🛠️ Skill gap analysis — detects matched and missing skills from a curated skill database
🤖 AI-generated feedback — Google Gemini API gives personalized resume improvement and interview prep suggestions
🎯 Overall match score & match level (Excellent / Good / Moderate / Low)
Tech Stack
`Python` · `scikit-learn` · `Sentence-Transformers` · `Streamlit` · `Google Gemini API` · `pypdf`
How It Works
Upload your resume (PDF)
Paste the job description
The app extracts resume text, predicts your resume category, computes semantic + skill match scores, and generates AI feedback
Run Locally
```bash
git clone https://github.com/<your-username>/ai-resume-job-matcher.git
cd ai-resume-job-matcher
pip install -r requirements.txt

# Add your Gemini API key
export GEMINI_API_KEY="your_key_here"

streamlit run app.py
```
Project Structure
```
├── app.py                       # Streamlit application
├── requirements.txt             # Python dependencies
└── models/
    ├── tfidf_vectorizer.pkl     # Trained TF-IDF vectorizer
    └── resume_classifier.pkl    # Trained resume category classifier
```
Author
Built by Amit Kumar Pandey as an AI/ML project combining classical ML, embeddings-based semantic search, and LLM-powered feedback generation.
