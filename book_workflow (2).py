import asyncio
import os
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import chromadb
import streamlit as st
import random
import string
from datetime import datetime
import numpy as np
from transformers import pipeline
import torch

# Initializing Hugging Face BART here, for spinning and reviewing
try:
    paraphraser = pipeline("text2text-generation", model="facebook/bart-large", device=0 if torch.cuda.is_available() else -1)
    st.write("Debug: Hugging Face BART model loaded successfully")
except Exception as e:
    st.error(f"Debug: Failed to load BART model: {e}")
    paraphraser = None

# for Web Scraping with Playwright
async def scrape_content_and_screenshot(url, output_dir="screenshots"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)
            
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            # targating the main content div (Wikisource-specific)
            content_div = soup.find('div', id='mw-content-text')
            if not content_div:
                st.error("Debug: Could not find main content div")
                return None, None
            # extracting text from the chapter content, excluding irrelevant text
            chapter_content = content_div.get_text(separator=' ', strip=True)
            
            screenshot_path = f"{output_dir}/screenshot_{datetime.now().strftime('%d%m%Y_%H%M%S')}.png"
            await page.screenshot(path=screenshot_path)
            
            await browser.close()
            st.write("Debug: Scraping completed successfully")
            return chapter_content, screenshot_path
    except Exception as e:
        st.error(f"Scraping failed: {e}")
        return None, None

# ai spinning with BART
def spin_content(text):
    if not text:
        st.error("Debug: spin_content: No input text provided")
        return None
    if not paraphraser:
        st.error("Debug: spin_content: BART model not loaded, using placeholder")
        return f"Placeholder: Rephrased version of {text[:300]}"
    
    try:
        # cutting text into chunks of ~300-char pieces for BART
        chunk_size = 300
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        spun_chunks = []
        for i, chunk in enumerate(chunks):
            prompt = f"Paraphrase the following text while maintaining its meaning and tone:\n\n{chunk}"
            result = paraphraser(prompt, max_length=300, num_return_sequences=1)
            spun_text = result[0]['generated_text']
            spun_chunks.append(spun_text)
            st.write(f"Debug: Spun chunk {i+1}/{len(chunks)}")
        spun_content = " ".join(spun_chunks)
        st.write("Debug: Spinning completed successfully with BART")
        return spun_content
    except Exception as e:
        st.error(f"Debug: Spinning failed: {e}, using placeholder")
        return f"Placeholder: Rephrased version of {text[:300]}"

# ai reviewing with BART
def review_content(text):
    if not text:
        st.error("Debug: review_content: No input text provided")
        return None
    if not paraphraser:
        st.error("Debug: review_content: BART model not loaded, using placeholder")
        return f"Placeholder: Review of {text[:300]} - Content is clear but could use more detail."
    
    try:
        # cutting text into chunks of ~300-char pieces for BART
        chunk_size = 300
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        review_chunks = []
        for i, chunk in enumerate(chunks):
            prompt = f"Review the following text for clarity, coherence, and quality. Provide a brief assessment and suggest improvements:\n\n{chunk}"
            result = paraphraser(prompt, max_length=200, num_return_sequences=1)
            review_text = result[0]['generated_text']
            review_chunks.append(review_text)
            st.write(f"Debug: Reviewed chunk {i+1}/{len(chunks)}")
        review_content = " ".join(review_chunks)
        st.write("Debug: Reviewing completed successfully with BART")
        return review_content
    except Exception as e:
        st.error(f"Debug: Reviewing failed: {e}, using placeholder")
        return f"Placeholder: Review of {text[:300]} - Content is clear but could use more detail."

# chromaDB for Versioning
def save_to_chromadb(content, version_id, collection_name="book_versions"):
    if not content:
        st.error("Debug: save_to_chromadb: No content to save")
        return None
    try:
        client = chromadb.Client()
        collection = client.create_collection(collection_name) if collection_name not in [c.name for c in client.list_collections()] else client.get_collection(collection_name)
        collection.add(documents=[content], ids=[version_id])
        st.write(f"Debug: ChromaDB saved version {version_id}")
        return version_id
    except Exception as e:
        st.error(f"Debug: ChromaDB save failed: {e}")
        return None

# RL-based Search with Q-learning
class QLSearch:
    def __init__(self, collection_name="book_versions"):
        try:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(collection_name)
            self.q_table = {}  # State-action value table
            self.alpha = 0.1  # Learning rate
            self.gamma = 0.9  # Discount factor
            self.epsilon = 0.1  # Exploration rate
        except Exception as e:
            st.error(f"Debug: ChromaDB init failed: {e}")

    def get_state(self, query):
        return hash(query) % 100

    def get_actions(self):
        return self.collection.get()['ids']

    def get_reward(self, query, document):
        try:
            results = self.collection.query(query_texts=[query], n_results=1)
            if results['documents']:
                return 1.0 - results['distances'][0][0]
            return 0.0
        except Exception:
            return 0.0

    def choose_action(self, state):
        actions = self.get_actions()
        if not actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_values = [self.q_table.get((state, a), 0.0) for a in actions]
        return actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table.get((state, action), 0.0)
        next_max = max([self.q_table.get((next_state, a), 0.0) for a in self.get_actions()], default=0.0)
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[(state, action)] = new_value

    def search(self, query):
        if not query:
            st.error("Debug: RL search: No query provided")
            return None
        try:
            state = self.get_state(query)
            for _ in range(10):
                action = self.choose_action(state)
                if not action:
                    return None
                document = self.collection.get(ids=[action])['documents'][0]
                reward = self.get_reward(query, document)
                next_state = self.get_state(document)
                self.update_q_table(state, action, reward, next_state)
                if reward > 0.4:
                    return document
                state = next_state
            results = self.collection.query(query_texts=[query], n_results=1)
            return results['documents'][0][0] if results['documents'] else None
        except Exception as e:
            st.error(f"Debug: RL search failed: {e}")
            return None

# Human-in-the-Loop Interface with Streamlit
def human_review_interface(spun_content, version_id):
    st.title("Book Publication Workflow")
    st.write("Review and Edit Spun Content")
    edited_content = st.text_area("Edit Content", spun_content if spun_content else "No spun content available", height=300)
    
    if st.button("Submit Feedback"):
        feedback = st.text_input("Feedback (e.g., 'Looks good' or 'Needs more clarity')")
        if feedback:
            new_version_id = f"v_{''.join(random.choices(string.ascii_lowercase, k=8))}"
            if save_to_chromadb(edited_content, new_version_id):
                st.success(f"Saved as version {new_version_id}")
                return edited_content, new_version_id
    return spun_content, version_id

# main workflow
async def main():
    st.set_page_config(page_title="Book Workflow")
    url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
    
    # 1. Scrape content
    with st.spinner("Scraping content..."):
        content, screenshot_path = await scrape_content_and_screenshot(url)
        if not content or not screenshot_path:
            st.error("Scraping failed, stopping workflow")
            return
        st.image(screenshot_path, caption="Screenshot of Source Page")
        st.write("Debug: Scraped content length:", len(content))

    # 2. AI Spin with BART
    spun_content = None
    with st.spinner("Spinning content with LLM..."):
        if content:
            spun_content = spin_content(content)
            if not spun_content:
                st.error("Spinning skipped due to LLM error")
                spun_content = content[:300]
            st.write("Debug: Spun Content:", spun_content)
        else:
            st.error("No content to spin, using placeholder")
            spun_content = "Placeholder content due to scraping failure"
    
    # 3. AI Review with BART
    with st.spinner("Reviewing content with LLM..."):
        review_result = review_content(spun_content)
        if review_result:
            st.write("Debug: AI Review:", review_result)
        else:
            st.error("AI Review skipped due to earlier error")
    
    # 4. Human Review
    version_id = f"v_{''.join(random.choices(string.ascii_lowercase, k=8))}"
    if save_to_chromadb(spun_content, version_id):
        st.success(f"Initial version saved as {version_id}")
    edited_content, new_version_id = human_review_interface(spun_content, version_id)
    
    # 5. Retrieve with RL Search
    if st.button("Retrieve Latest Version"):
        with st.spinner("Searching with RL algorithm..."):
            rl_search = QLSearch()
            retrieved_content = rl_search.search(edited_content)
            if retrieved_content:
                st.write("Debug: Retrieved Content:", retrieved_content)
            else:
                st.error("Retrieval failed")

# Run the app in Colab's event loop
if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(main())
    except RuntimeError:
        asyncio.run(main())
