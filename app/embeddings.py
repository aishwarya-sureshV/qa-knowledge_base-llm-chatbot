import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print('WARNING: OPENAI_API_KEY not set. Embedding calls will fail until you set it.')

# Singleton client - no need to recreate it every time
_client = None

def get_client():
    # Reuse the same client instance if we already created one
    # Saves a tiny bit of overhead, plus it's cleaner
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before using embeddings.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def get_embeddings(texts):
    # Batch the embedding requests - OpenAI lets us do up to 2048 items per request
    # But 16 at a time keeps it reasonable and avoids timeouts
    # Returns a list of vectors (1536 dimensions each for text-embedding-3-small)
    client = get_client()
    results = []
    BATCH = 16  # Tried 32 but got rate limited, 16 seems safer
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model='text-embedding-3-small', input=batch)
        for item in resp.data:
            results.append(item.embedding)
    return results
