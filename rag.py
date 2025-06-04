import re
import string
import requests

def clean_text(text: str) -> str:
    text = text.strip()
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))  
    return text

def classify_query(query: str) -> str:
    cleaned = clean_text(query)
    prompt = f"""
Classify the following user query. Does it need real-time (latest news, updates) or static (general facts) info? 

Or is it related to code, math, search, or tools like calendar or news?

Query: "{cleaned}"

Reply only with one of these labels:
- real-time
- static
- code
- math
- search
- tool:calendar
- tool:news
"""
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"].strip().lower()

if __name__ == "__main__":
    user_input = input("Enter your query: ")
    print(f"Processing query: '{user_input}'")
    label = classify_query(user_input)
    print("Classification result:", label)
