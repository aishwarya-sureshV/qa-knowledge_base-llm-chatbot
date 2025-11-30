import os, uuid, json
import pdfplumber

# Chunk size - 400 words works pretty well, not too big not too small
# Tried 200 and 800, 400 seems like the sweet spot
CHUNK_SIZE_TOKENS = 400
CHROMA_DIR = os.path.join(os.getcwd(), 'chroma_db')  # Where we store the vector DB

def _read_txt(path):
    # Simple text file reader - utf-8 with ignore for weird characters
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def _read_pdf(path):
    # PDF reading is tricky - pdfplumber is pretty reliable though
    text = ''
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                text += (p.extract_text() or '') + '\n'  # Some pages might be empty
    except Exception as e:
        print('pdf read error', e)  # Just print it, keep going with what we have
    return text

def load_document(path):
    # Quick file type check and route to the right reader
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.txt', '.md'):
        return _read_txt(path)
    if ext in ('.pdf',):
        return _read_pdf(path)
    # Default to txt if we don't recognize it
    return _read_txt(path)

def chunk_text(text, chunk_size=CHUNK_SIZE_TOKENS):
    # Super simple chunking - just split by words
    # Could be smarter about sentence boundaries but this works fine for now
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_file(path):
    # Load the doc, chunk it up, save each chunk as a separate JSON file
    # Using UUIDs so we don't have naming conflicts
    text = load_document(path)
    chunks = chunk_text(text)
    saved = 0
    out_dir = os.path.join(os.getcwd(), 'data_chunks')
    os.makedirs(out_dir, exist_ok=True)
    for c in chunks:
        uid = str(uuid.uuid4())
        with open(os.path.join(out_dir, uid + '.json'), 'w', encoding='utf-8') as f:
            json.dump({'text': c}, f)
        saved += 1
    return saved  # Return count so caller knows how many chunks were created

def load_all_chunks():
    # Load all the JSON chunk files we created earlier
    # Sorted to keep order consistent (though it shouldn't matter)
    out_dir = os.path.join(os.getcwd(), 'data_chunks')
    docs = []
    if not os.path.exists(out_dir):
        return docs  # Empty list if no chunks yet
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith('.json'):
            with open(os.path.join(out_dir, fn), 'r', encoding='utf-8') as f:
                docs.append(json.load(f)['text'])
    return docs

def build_collection(persist=False):
    # This is the heavy lifting - creates the vector DB from all chunks
    import chromadb
    if persist:
        # Persistent means it saves to disk - use this for production
        client = chromadb.PersistentClient(path=CHROMA_DIR)
    else:
        # In-memory only, gone after process ends
        client = chromadb.Client()
    collection = client.get_or_create_collection(name="qa_knowledge")
    docs = load_all_chunks()
    if len(docs) == 0:
        return collection  # Empty collection if no docs yet
    
    # Generate embeddings - this is the expensive part (API calls)
    from .embeddings import get_embeddings
    embs = get_embeddings(docs)
    
    # Simple IDs and metadata - could track source files here if needed
    ids = [str(i) for i in range(len(docs))]
    metadata = [{'source': 'local', 'index': i} for i in range(len(docs))]
    collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metadata)
    return collection
