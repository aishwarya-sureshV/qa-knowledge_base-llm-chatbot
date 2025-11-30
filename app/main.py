from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
from .ingest import ingest_file, build_collection
from .qa import answer_query

# Load env vars - gotta have that API key somewhere
load_dotenv()

app = FastAPI(title="QA Knowledge Base - MVP")


@app.post('/ingest')
async def ingest(file: UploadFile = File(...)):
    # User uploads a file, we save it to /tmp first
    # TODO: maybe clean up /tmp files after processing? nah, OS will handle it
    contents = await file.read()
    tmp_path = os.path.join('/tmp', file.filename)
    with open(tmp_path, 'wb') as f:
        f.write(contents)
    # This does the actual chunking and saving to data_chunks/
    n = ingest_file(tmp_path)
    return JSONResponse({'status':'ok', 'indexed_chunks': n})


@app.post('/build')
async def build():
    # This is where the magic happens - takes all chunks and builds the vector DB
    # persist=True means it'll stick around after server restarts
    c = build_collection(persist=True)
    # getattr is here cuz ChromaDB API changed and name might not exist
    return JSONResponse({'status':'ok', 'collection': getattr(c, "name", "qa_knowledge")})


class Query(BaseModel):
    q: str
    conversation_history: list = []  # List of {role, content} messages - empty by default

@app.post('/query')
async def query(q: Query):
    try:
        # This is the main RAG flow - retrieve docs, build context, ask LLM
        answer = answer_query(q.q, conversation_history=q.conversation_history)
        return JSONResponse(answer)
    except ValueError as e:
        # Usually means collection doesn't exist - user needs to run /build first
        return JSONResponse(status_code=400, content={'error': str(e), 'type': 'ValidationError'})
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        
        # OpenAI errors are super cryptic, so let's make them actually readable
        if 'RateLimitError' in error_type or '429' in error_msg or 'quota' in error_msg.lower():
            user_msg = "OpenAI API quota exceeded. Please check your API key and billing details, or update the OPENAI_API_KEY in your .env file."
        elif 'API key' in error_msg or 'authentication' in error_msg.lower():
            user_msg = "OpenAI API key issue. Please check your OPENAI_API_KEY in the .env file."
        else:
            user_msg = error_msg
        
        error_detail = {
            'error': user_msg,
            'type': error_type,
            'details': error_msg  # Keep the ugly details for debugging
        }
        return JSONResponse(status_code=500, content=error_detail)


if __name__ == '__main__':
    # Run the server - default port 8000, accessible from anywhere (0.0.0.0)
    uvicorn.run(app, host='0.0.0.0', port=8000)
