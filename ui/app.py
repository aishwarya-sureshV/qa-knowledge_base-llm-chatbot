import streamlit as st
import requests

st.set_page_config(page_title='QA Knowledge Base - MVP', page_icon='🤖', layout='wide')
st.title('QA knowledge base - MVP')

# Backend API - configurable via environment variable for deployment
import os
api_base = os.environ.get('API_BASE_URL', 'http://localhost:8000')

# Store conversation in session state so it persists across reruns
# Streamlit reruns the whole script on every interaction, so we need this
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

# Show the conversation so far
if st.session_state['conversation_history']:
    for i, msg in enumerate(st.session_state['conversation_history']):
        with st.chat_message(msg['role']):
            st.write(msg['content'])
            # Could show sources here but it gets cluttered, so we only show for latest response
            if msg['role'] == 'assistant' and i < len(st.session_state['conversation_history']) - 1:
                pass
else:
    st.info('👋 Start by asking a question below!')

# Main chat input - this triggers when user types and hits enter
if prompt := st.chat_input("Ask a question..."):
    # Save user's question to history first
    st.session_state['conversation_history'].append({
        'role': 'user',
        'content': prompt
    })
    
    # Show it immediately so UI feels responsive
    with st.chat_message('user'):
        st.write(prompt)
    
    # Send previous messages (not the current one) to the API
    # API will add the current question itself with context
    conversation_for_api = [
        msg for msg in st.session_state['conversation_history'][:-1] 
        if msg['role'] in ['user', 'assistant']
    ]
    
    # Call the backend and show the response
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            try:
                payload = {
                    'q': prompt,
                    'conversation_history': conversation_for_api
                }
                r = requests.post(f'{api_base}/query', json=payload, timeout=60)
                
                if r.status_code == 200:
                    data = r.json()
                    answer = data.get('answer', 'No answer provided')
                    
                    # Show the answer
                    st.write(answer)
                    
                    # Save it to history so we can reference it later
                    st.session_state['conversation_history'].append({
                        'role': 'assistant',
                        'content': answer
                    })
                    
                    # Show source documents in a collapsible section
                    sources = data.get('sources', [])
                    if sources:
                        with st.expander(f"📚 View {len(sources)} source(s)", expanded=False):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**")
                                # Truncate long sources - no one wants to read a whole page
                                st.text(source[:500] + '...' if len(source) > 500 else source)
                                if i < len(sources):
                                    st.markdown('---')
                else:
                    # API error - show it and remove the failed user message
                    error_msg = f'Error: {r.status_code} {r.text}'
                    st.error(error_msg)
                    st.session_state['conversation_history'].pop()
            except Exception as e:
                # Network error or something else went wrong
                error_msg = f'Error: {e}'
                st.error(error_msg)
                st.session_state['conversation_history'].pop()
    
    # Rerun to update the UI with the new message
    st.rerun()
