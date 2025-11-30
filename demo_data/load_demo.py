import os, textwrap
from dotenv import load_dotenv
from app.ingest import ingest_file, build_collection

# Need env vars for the API key
load_dotenv()
os.makedirs('demo_data', exist_ok=True)

# Look for any PDFs, txt, or markdown files in demo_data folder
# Chunk them up and save to data_chunks/
folder = "demo_data"
for f in os.listdir(folder):
    if f.lower().endswith((".pdf", ".txt", ".md")):
        print("Ingesting:", f)
        ingest_file(os.path.join(folder, f))

# Now build the vector database from all those chunks
# This is where the expensive embedding API calls happen
print('Created demo docs and ingested chunks. Now build collection...')
build_collection(persist=True)
print('Done. Collection persisted.')
