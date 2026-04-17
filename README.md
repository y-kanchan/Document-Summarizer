# 📄 DocumentLens — NLP Document Summarizer

A self-contained, no-Docker Python web application that summarises long documents using **extractive NLP** techniques (TF-IDF, positional scoring, sentence length analysis) built on **Flask + NLTK**.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Extractive Summarisation** | Multi-signal scoring: TF-IDF (60%) + Position (20%) + Length (20%) |
| **Keyword Extraction** | Top-N key terms highlighted visually |
| **Text Analytics** | Word count, sentence count, lexical diversity, Flesch Reading Ease |
| **Adjustable Length** | Slider to control summary ratio (10% – 60%) |
| **3 Sample Texts** | Science, Technology, History — try instantly |
| **Download & Copy** | One-click copy or .txt download of the summary |
| **Keyboard Shortcut** | `Ctrl+Enter` (or `Cmd+Enter`) to summarise |
| **No Docker** | Pure Python + venv — no containers needed |

---

## 📋 Requirements

- **Python 3.9 or higher** (3.10+ recommended)
- No Docker, no database, no cloud services needed

Check your version:
```bash
python --version
# or
python3 --version
```

Download Python from: https://www.python.org/downloads/

---

## 🚀 Quick Start

### Option A — One-click script (Recommended)

**macOS / Linux:**
```bash
# 1. Make the script executable (only needed once)
chmod +x start.sh

# 2. Run it
./start.sh
```

**Windows:**
```cmd
# Double-click start.bat  — OR — in Command Prompt:
start.bat
```

The script will:
1. Find your Python installation
2. Create an isolated virtual environment (`venv/`)
3. Install all dependencies automatically
4. Start the server at **http://127.0.0.1:5000**

---

### Option B — Manual setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
#    macOS/Linux:
source venv/bin/activate
#    Windows CMD:
venv\Scripts\activate.bat
#    Windows PowerShell:
venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## 📁 Project Structure

```
doc-summarizer/
│
├── app.py              ← Flask server + all NLP logic
├── requirements.txt    ← Python dependencies (Flask, NLTK)
├── start.sh            ← One-click launcher (macOS/Linux)
├── start.bat           ← One-click launcher (Windows)
├── README.md           ← This guide
│
└── templates/
    └── index.html      ← Complete frontend (HTML/CSS/JS)
```

---

## 🔌 REST API Endpoints

The app also exposes a JSON API you can use programmatically:

### `POST /api/summarize`
Summarise a document.

**Request body:**
```json
{
  "text": "Your long document text here...",
  "ratio": 0.3
}
```

**Response:**
```json
{
  "success": true,
  "summary": "Extracted summary sentences...",
  "stats": {
    "word_count": 450,
    "sentence_count": 22,
    "unique_words": 210,
    "avg_sentence_length": 20.5,
    "lexical_diversity": 0.467,
    "flesch_reading_ease": 58.3,
    "readability_level": "Moderate",
    "char_count": 2800
  },
  "summary_stats": {
    "original_sentences": 22,
    "summary_sentences": 7,
    "compression_ratio": 0.31,
    "method": "extractive (TF-IDF + Position + Length)"
  },
  "keywords": ["machine", "learning", "data", "neural", "model", ...]
}
```

### `POST /api/keywords`
Extract keywords only.
```json
{ "text": "...", "top_n": 15 }
```

### `POST /api/analyze`
Get text statistics only.
```json
{ "text": "..." }
```

---

## 🧠 How the NLP Works

The summariser uses three signals combined in a weighted score:

```
Final Score = 0.60 × TF-IDF Score
            + 0.20 × Position Score
            + 0.20 × Length Score
```

**1. TF-IDF Score (60% weight)**
- Counts word frequency across the document (Term Frequency)
- Penalises words that appear in too many sentences (Inverse Document Frequency)
- Sentences with high-value rare terms score higher

**2. Position Score (20% weight)**
- Sentences in the first 20% of the document score higher (topic introduction)
- Sentences in the last 10% score slightly higher (conclusion)
- Middle sentences score lower

**3. Length Score (20% weight)**
- Optimal sentence length: 10–35 words scores 1.0
- Very short (<10 words) or very long (>35 words) sentences score lower

Top-scoring sentences are selected and returned **in their original order** to preserve readability.

---

## 🛠 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` | Make sure your venv is activated, then `pip install -r requirements.txt` |
| `Address already in use` (port 5000) | Change port in `app.py`: `app.run(port=5001)` |
| NLTK data download errors | Run `python -c "import nltk; nltk.download('all')"` once |
| Windows: `venv\Scripts\activate` fails | Run `Set-ExecutionPolicy RemoteSigned` in PowerShell as admin |
| macOS port 5000 conflict (AirPlay) | Change port to 5001 in `app.py` |

---

## 🔧 Configuration

Edit `app.py` to change:
- **Port**: `app.run(port=5001)` (line near the bottom)
- **Max text length**: `50000` characters (search for `50000`)
- **Score weights**: `0.60`, `0.20`, `0.20` in `extractive_summarize()`
- **Debug mode**: `app.run(debug=False)` for production

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| Flask | ≥3.0.0 | Web server & API |
| NLTK | ≥3.8.1 | Tokenisation, stopwords |

Total install size: ~50 MB (mostly NLTK data)

---

## 🔒 Privacy

All processing happens **locally on your machine**. No text is sent to any external service or API.

---

*Built with Python · Flask · NLTK · No Docker required*
