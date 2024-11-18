from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


import streamlit as st
import torch

set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
)


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="index")

# load index
index = load_index_from_storage(storage_context, embed_model= embed_model)



model_url = "https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/resolve/main/capybarahermes-2.5-mistral-7b.Q4_0.gguf"


llm = LlamaCPP(
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={
        "n_gpu_layers": -1,
        "torch_dtype": torch.float16,
        "load_in_8bit": True,
    },
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)



st.header("Chat with LOTR 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about JRRT's LOTR"}
    ]



@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the LOTR – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data/lotr-chap-1-3/", recursive=True)
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        return index

index = load_data()

# set up query engine
query_engine = index.as_query_engine(llm=llm, chat_mode="condense_question", verbose=True)


if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history