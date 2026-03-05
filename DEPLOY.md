# Render Deployment Guide — Deep Research Engine

## ⚠️ Cold Start Disclaimer

> **Render Free Tier services spin down after 15 minutes of inactivity.** The first request after idle will take **30–60 seconds** as the container restarts. Deep research queries may take additional time (30–120s) due to multi-step web scraping and LLM synthesis.

---

## 🔑 Getting Your Free Gemini API Key

1. Visit **[Google AI Studio](https://aistudio.google.com/app/apikey)**
2. Sign in with your Google account (Gmail works)
3. Click **"Create API Key"** → Select or create a Google Cloud project
4. Copy the key — **completely free**, no credit card required

### Free Tier Rate Limits

| Model | Requests/Min | Tokens/Min |
|-------|-------------|------------|
| `gemini-2.0-flash-thinking-exp-01-21` | 10 RPM | 4M TPM |
| `gemini-2.5-pro` | 2 RPM | 1M TPM |
| `gemini-flash-latest` | 15 RPM | 1M TPM |

### Google Custom Search Engine ID (Optional)
For web search functionality:
1. Go to [Programmable Search Engine](https://programmablesearchengine.google.com/)
2. Create a search engine → Get the **CX ID**
3. The Custom Search JSON API provides **100 free queries/day**

---

## 🔄 Model Fallback Routing

The app currently uses `gemini-2.0-flash-thinking-exp-01-21`. For production resilience, the recommended fallback chain is:

| Priority | Model | Use Case |
|----------|-------|----------|
| 1 | `gemini-2.5-pro` | Best quality research synthesis |
| 2 | `gemini-2.0-flash-thinking-exp-01-21` | Good quality with reasoning |
| 3 | `gemini-flash-latest` | Fast fallback for rate-limited scenarios |

> **Implementation:** The model name is configurable via the `ResearchConfig.model_name` field. For autonomous fallback, wrap the `model.generate_content()` calls with a try/except that catches `429` or `ResourceExhausted` errors and retries with the next model in the chain.

---

## Environment Variables (Render Dashboard)

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | Gemini API key from [AI Studio](https://aistudio.google.com/app/apikey) |
| `SEARCH_ENGINE_ID` | Optional | Google Custom Search Engine CX ID |
| `GEMINI_API_KEY` | Optional | Alternative key name (app checks both) |

---

## Deployment Steps

### 1. Render Web Service
1. Go to [render.com/new](https://dashboard.render.com/new)
2. Connect GitHub repo, set **Root Directory** to the project folder
3. Select **Docker** environment → **Free** instance type
4. Add environment variables in the Render dashboard

### 2. Configuration Files
```
Dockerfile → python:3.10-slim → pip install requirements + gunicorn
start.sh   → gunicorn app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker
```

---

## Resource Limits

| Resource | Render Free Tier | This Project |
|----------|-----------------|--------------|
| RAM | 512 MB | ~120 MB (FastAPI + httpx + sqlite) |
| Storage | 1 GB | ~1 MB code + sqlite DB grows over time |
| Bandwidth | 100 GB/month | Moderate (web scraping + API calls) |
