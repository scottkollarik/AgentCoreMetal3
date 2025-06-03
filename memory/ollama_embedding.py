import requests

def get_ollama_embedding(text, model="mxbai-embed-large"):
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raises an error if the request failed
    return response.json()["embedding"]  # Returns the embedding as a list of floats