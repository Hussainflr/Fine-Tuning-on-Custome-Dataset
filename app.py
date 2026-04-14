import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread



# Load Model (cached)

@st.cache_resource
def load_model():
    model_path = "outputs/final_model"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    return model, tokenizer


model, tokenizer = load_model()



st.sidebar.title("⚙️ Settings")

temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9)
max_tokens = st.sidebar.slider("Max Tokens", 50, 512, 200)



if st.sidebar.button("🧹 Reset Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    st.rerun()



def stream_response(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    return streamer


# Streamlit UI

st.set_page_config(page_title="Qwen Chat", page_icon="🤖")
st.title("🤖 Qwen Fine-Tuned Chatbot")

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

# Display history
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response (streaming)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        streamer = stream_response(st.session_state.messages)

        for new_text in streamer:
            full_response += new_text
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    # Save response
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )