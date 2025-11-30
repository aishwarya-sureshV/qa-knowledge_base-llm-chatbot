import os
from .ingest import CHROMA_DIR
import chromadb
from .embeddings import get_embeddings, get_client
from dotenv import load_dotenv

load_dotenv()

# Can override this in .env if you want to use a different model
# gpt-4o-mini is cheap and fast, good enough for this use case
OPENAI_CHAT_MODEL = os.environ.get('OPENAI_CHAT_MODEL', 'gpt-4o-mini')

# Another singleton - just reuse the embeddings client since it's the same API
_openai_client = None

def get_openai_client():
    # Reuse the embeddings client, it's the same OpenAI API anyway
    global _openai_client
    if _openai_client is None:
        from .embeddings import get_client
        _openai_client = get_client()
    return _openai_client

def get_collection():
    # Connect to ChromaDB and grab our collection
    # If it doesn't exist, user needs to run /build first
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection('qa_knowledge')
    except Exception as e:
        raise ValueError(f"Collection 'qa_knowledge' not found. Please build the collection first using /build endpoint. Error: {e}")

def retrieve(query):
    # The retrieval part of RAG - find similar chunks based on the query
    coll = get_collection()
    q_emb = get_embeddings([query])[0]  # Embed the user's question
    
    # Limit to 10 docs - learned the hard way that more = token limit errors
    # Could make this configurable but 10 seems to work well
    MAX_RESULTS = 10
    count = coll.count()
    n_results = min(MAX_RESULTS, count) if count > 0 else 10
    
    res = coll.query(query_embeddings=[q_emb], n_results=n_results)
    docs = []
    
    # ChromaDB changed their response format at some point, so we handle both
    # This ugly code is why - compatibility with different versions
    if isinstance(res, dict):
        # Old format - dict with 'documents' key
        if 'documents' in res and len(res['documents']) > 0:
            for d in res['documents'][0]:
                docs.append(d)
    else:
        # New format - object with .documents attribute
        documents = getattr(res, 'documents', None)
        if documents and len(documents) > 0:
            for d in documents[0]:
                docs.append(d)
        elif hasattr(res, '__getitem__'):
            # Last resort - try dict-like access
            try:
                for d in res['documents'][0]:
                    docs.append(d)
            except (KeyError, TypeError):
                pass
    return docs

def assemble_prompt(question, docs):
    # Old prompt function - not used anymore but keeping it around just in case
    # We switched to the chat API format instead
    ctx = "\n\n---\n\n".join(docs)
    prompt = f"You are an expert QA assistant. Use the context below to answer the question as concisely as possible. If the information is not available, say you don't know.\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    return prompt

def call_openai(messages, max_tokens=512):
    # Actually call OpenAI chat API - the generation part of RAG
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,  # Keep it deterministic - no randomness needed for Q&A
    )
    return resp.choices[0].message.content

def truncate_conversation(conversation_history, max_messages=10):
    # Trim conversation history if it's getting too long
    # Token limits are real and they will bite you if you ignore them
    if len(conversation_history) <= max_messages:
        return conversation_history
    
    # If there's a system message, keep it and just trim the rest
    # Otherwise just take the most recent messages
    if len(conversation_history) > 0 and conversation_history[0].get('role') == 'system':
        return [conversation_history[0]] + conversation_history[-(max_messages-1):]
    return conversation_history[-max_messages:]

def answer_query(question, conversation_history=None):
    # The main RAG function - retrieve, augment, generate
    # conversation_history lets us have actual conversations, not just one-off Q&A
    if conversation_history is None:
        conversation_history = []
    
    # Step 1: Find relevant document chunks based on the question
    docs = retrieve(question)
    
    # Step 2: Build the prompt with retrieved context
    ctx = "\n\n---\n\n".join(docs)
    context_message = f"Context from documents:\n{ctx}\n\nQuestion: {question}"
    
    # Step 3: Build the message list for chat API (includes conversation history)
    messages = []
    
    # System prompt tells the model how to behave
    messages.append({
        'role': 'system',
        'content': 'You are an expert QA assistant. Use the context provided to answer questions. If information is not available in the context, say you don\'t know. You can reference previous parts of the conversation when answering follow-up questions.'
    })
    
    # Add previous conversation turns (but trim if too long)
    if conversation_history:
        # Don't include system messages from history - we add our own
        history_messages = [msg for msg in conversation_history if msg.get('role') != 'system']
        messages.extend(truncate_conversation(history_messages, max_messages=8))
    
    # Add the current question with retrieved context
    messages.append({
        'role': 'user',
        'content': context_message
    })
    
    # Step 4: Ask the LLM to generate an answer
    answer = call_openai(messages, max_tokens=512)
    
    # Return both the answer and the sources (for citations/transparency)
    return {'answer': answer, 'sources': docs}
