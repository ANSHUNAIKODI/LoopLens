# LoopLens 🔍

> **"Your time is valuable. Check before you watch."**

LoopLens is a real, functional web application built to help YouTube viewers identify videos that may contain repeated footage, loop-padded sections to manipulate ad revenue, or cut-off/incomplete uploads.

---

## 🌟 Key Features

1. **Dark Navy & Mint-Green Aesthetic:**
   - Modern, responsive frontend built with semantic HTML5, Vanilla CSS3 tokens, glassmorphism, glowing accents, and Outfit/Plus Jakarta Sans typography.
2. **Authentic Data Only (Zero Fabrication):**
   - Strictly queries the real **YouTube Data API v3**.
   - Never invents video details, fake comments, or manufactured ratings.
   - If an API key is missing or comments are disabled, LoopLens displays clear, informative explanations.
3. **Evidence-Based Community Analysis:**
   - Inspects real viewer comments for authentic community reports of duplicated footage, loops, padding, or premature cut-offs.
   - Highlights matching comment quotes and surfaces mentioned timestamps (e.g. `03:45`).
   - Clearly distinguishes community reports from pixel-level computer vision video verification.
4. **Secure Server-Side Credentials:**
   - `YOUTUBE_API_KEY` is loaded from the server's `.env` environment file.
   - The key is **never** sent to or exposed in frontend client code.

---

## 🏗️ Project Architecture

```
LoopLens/
├── backend/
│   ├── __init__.py
│   ├── config.py           # Loads .env & handles API key securely
│   ├── youtube_service.py  # Real YouTube Data API v3 metadata & comments fetcher
│   ├── analyzer.py         # Heuristic pattern & evidence analyzer for comments
│   └── main.py             # FastAPI app, REST endpoints, and static file mounting
├── frontend/
│   ├── index.html          # Responsive semantic markup & structure
│   ├── styles.css          # Dark navy + mint green theme with glassmorphism
│   └── app.js              # Client logic, URL validation, API requests, dynamic UI
├── tests/
│   ├── test_backend.py     # Unit tests for URL parser, ISO duration, and analyzer
│   └── test_api.py         # Integration tests for FastAPI endpoints and static assets
├── .env.example            # Template for environment variables
├── .env                    # Local environment file (stores YOUTUBE_API_KEY)
├── requirements.txt        # Python package dependencies
├── run.py                  # Server launcher script
└── README.md               # Documentation & setup guide
```

---

## 🚀 Quickstart Guide

### 1. Requirements
- Python 3.10+ (Tested on Python 3.13)
- An active YouTube Data API v3 key (free from Google Cloud Console)

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure Your YouTube API Key
Open `.env` (or copy `.env.example` to `.env`) and add your API key:
```ini
YOUTUBE_API_KEY=AIzaSy...Your_Key_Here
PORT=8000
HOST=127.0.0.1
```

> **How to get a free YouTube API Key:**
> 1. Visit the [Google Cloud Console](https://console.cloud.google.com/).
> 2. Create a project and navigate to **APIs & Services > Library**.
> 3. Search for **YouTube Data API v3** and click **Enable**.
> 4. Go to **APIs & Services > Credentials** and click **Create Credentials > API Key**.
> 5. Paste the key into `.env`.

### 4. Run the Application
```powershell
python run.py
```
Or with Uvicorn directly:
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧪 Running Automated Tests

Run the complete test suite:
```powershell
python -m unittest discover -s tests
```
All 11 unit and API integration tests will run and validate URL parsing, ISO 8601 duration parsing, comment analysis logic, and error handling.

---

## 🛡️ Community Reports vs. Video Repetition Detection

LoopLens follows a strict honesty policy:
- **Community Report Analysis (Active):** Scans authentic public comments retrieved through the official YouTube API v3 to identify reports by fellow viewers who flagged repetition, looping, or truncated videos.
- **Computer-Vision Frame Verification (Future Milestone):** Directly downloading and comparing video frames/hashes to mathematically prove identical segments will be implemented as an opt-in processing pipeline in future releases.
