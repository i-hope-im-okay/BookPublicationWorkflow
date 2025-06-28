# Automated Book Publication Workflow
Internship project for Soft-Nerve. Scrapes a full chapter from Wikisource, uses Hugging Face's BART LLM to paraphrase and review text, supports human feedback via Streamlit, stores versions in ChromaDB, and retrieves using RL-based Q-learning.

## Setup
- Install Python packages individually: `playwright`, `beautifulsoup4`, `chromadb`, `streamlit`, `pyngrok`, `numpy`, `torch`, `transformers`.
- Run `playwright install` and `playwright install-deps`.
- Set `NGROK_AUTH_TOKEN` in Colab Secrets or enter manually.
- Execute `streamlit run book_workflow.py`.

## Demo
See `demo.mp4` for a walkthrough. 
Note: The xAI API (beta) returned 400/403/404 errors, and Gemini/OpenAI hit 429 rate limits, so Hugging Face's BART LLM was used for reliable, local performance. Scraping targets the full chapter, excluding irrelevant content.

## Features
- Scraping: Playwright extracts a full chapter’s text and screenshots, targeting the main content.
- Spinning/Reviewing: BART LLM paraphrases and reviews the full chapter in chunks.
- Versioning: ChromaDB stores content versions.
- Retrieval: RL-based Q-learning for version search.
- UI: Streamlit for human review and editing.
