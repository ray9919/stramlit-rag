"""
A simple web application to implement a chatbot. This app uses Streamlit 
for the UI and the Python requests package to talk to an API endpoint that
implements text generation and Retrieval Augmented Generation (RAG) using LLMs
and Amazon OpenSearch as the vector database.
"""
import base64
import json

import streamlit as st
import requests as req
from typing import List, Tuple, Dict
import uuid


def _get_session():
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session.id


# 生成唯一的会话 ID
session_id = _get_session()

print(f"session_id={session_id}")

outputs = {}

# global constants
STREAMLIT_SESSION_VARS: List[Tuple] = [("generated", []), ("past", []), ("input", ""), ("stored_session", [])]
HTTP_OK: int = 200

# two options for the chatbot, 1) get answer directly from the LLM
# 2) use RAG (find documents similar to the user query and then provide
# those as context to the LLM).
MODE_RAG: str = 'RAG'
MODE_TEXT2TEXT: str = 'Text Generation'
MODE_VALUES: List[str] = [MODE_RAG, MODE_TEXT2TEXT]

# Currently we use the flan-t5-xxl for text generation
# and gpt-j-6b for embeddings but in future we could support more

# if running this app on a compute environment that has
# IAM cloudformation::DescribeStacks access read the 
# stack outputs to get the name of the LLM endpoint

# create an outputs dictionary with keys of interest
# the key value would need to be edited manually before
# running this app

api_rag_ep: str = 'https://9ek0hrjut6.execute-api.us-east-1.amazonaws.com/chat_by_telsa_bot'

print(f"api_rag_ep={api_rag_ep}")

####################
# Streamlit code
####################

# Page title
st.set_page_config(page_title='特斯拉智能问答客服', layout='wide')

# keep track of conversations by using streamlit_session
_ = [st.session_state.setdefault(k, v) for k, v in STREAMLIT_SESSION_VARS]


# Define function to get user input
def get_user_input() -> str:
    """
    Returns the text entered by the user
    """
    print(st.session_state)
    input_text = st.text_input("You: ",
                               st.session_state["input"],
                               key="input",
                               placeholder="你可以问我，任何用车方面的问题",
                               label_visibility='hidden')

    return input_text


st.title("特斯拉智能问答客服")
st.subheader(
    f" Powered by AWS Bedrock with Claude3.0 and Titan Embeddings G1 - Text models for embeddings")

# get user input
user_input: str = get_user_input()
mode = MODE_RAG
# based on the selected mode type call the appropriate API endpoint
if user_input:
    # headers for request and response encoding, same for both endpoints
    headers: Dict = {"accept": "text/plain", "Content-Type": "text/plain"}
    output: str = None
    print(f'user_input = {user_input}')
    payload = {
        'user_input': user_input,
        'session_id': session_id
    }
    json_string = json.dumps(payload)
    base64data = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')
    print(f'base64data = {base64data}')
    resp = req.post(api_rag_ep, headers=headers, json=base64data)
    if resp.status_code != HTTP_OK:
        output = resp.text
    else:
        resp = resp.json()
        print(resp)
        # sources = [d['metadata']['source'] for d in resp['docs']]
        output = resp['answer']

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# download the chat history
download_str: List = []
with st.expander("历史对话", expanded=True):
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="❓")
        st.success(st.session_state["generated"][i], icon="👩‍💻")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download', download_str)
