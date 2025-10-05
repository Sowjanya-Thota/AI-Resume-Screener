
````markdown
# 🧠 Advanced Resume Screening System

A **production-like, single-file resume screening system** built with **Streamlit** for parsing, analyzing, and scoring resumes against job descriptions (JD). It combines NLP, TF-IDF, optional BERT embeddings, keyword extraction, and tech skill matching to provide a comprehensive candidate evaluation.

---

## Features

- **Section Parsing**: Extract Contact, Education, Experience, Skills, Summary.
- **Similarity Scoring**:
  - TF-IDF similarity between JD and resume
  - Optional BERT semantic similarity
  - Section-aware similarity (experience, skills, summary)
- **Keyword Analysis**:
  - Per-keyword explainability
  - JD keyword coverage
  - Tech skill matching
- **Experience Insights**:
  - Detect total years of experience
  - Employment gap detection
  - Structured experience parsing
- **Candidate Comparison**:
  - Duplicate/resume-similarity detection
  - Export individual candidate JSON
  - Dashboard with interactive charts & gauges
- **Customizable Weighting**:
  - Adjust weights for TF-IDF, BERT, skills, experience, tech match via sidebar
- **Multi-format Document Support**: PDF, DOCX, TXT
- **Graceful Fallback**: Optional heavy libraries (BERT, SpaCy) handled automatically

---

## Installation

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd <your-repo-folder>
````

2. **Create and activate a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> **Note:** For BERT embeddings, install `sentence-transformers`:
>
> ```bash
> pip install sentence-transformers
> ```

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Upload resumes (PDF, DOCX, TXT) and the job description.
3. Adjust weights for different scoring criteria in the sidebar.
4. View results:

   * Candidate similarity scores
   * Skill matching overview
   * Keyword coverage
   * Experience summary
5. Export individual candidate results to JSON for record-keeping.

---

## Technologies

* Python 3.10+
* Streamlit
* NLP: SpaCy (optional), TF-IDF, Sentence Transformers (optional BERT)
* PDF Parsing: PyPDF2
* DOCX Parsing: python-docx
* Data Analysis: pandas, scikit-learn
* Visualization: Streamlit charts, gauges

---

## Project Structure

```
resume-screening-system/
│
├─ app.py              # Main Streamlit application
├─ requirements.txt    # Python dependencies
├─ README.md           # Project documentation
└─ sample_resumes/     # Folder for testing resumes
```

---

## Customization

* **Weight Adjustments:** Change how TF-IDF, BERT, skills, experience, and tech match impact final scores.
* **Supported Formats:** Add more parsers if needed (e.g., DOC).
* **Visualization:** Add charts, gauges, or dashboards to highlight key metrics.
* **Keyword Matching:** Customize the skill/keyword dictionary to match your domain.

---

## Future Enhancements

* Add login and database persistence (PostgreSQL/MySQL + SQLAlchemy)
* Integrate email notifications for HR managers
* Auto-generate summary reports and dashboards
* Multi-language resume support
* Interactive candidate comparison table

---

## License

MIT License © 2025

