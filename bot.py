import streamlit as st
from utils import write_message
from agent import generate_response
# Page Config
st.set_page_config("Ebert", page_icon=":movie_camera:")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a 5G Chatbot for ETSI's NRUP Specifications! I am using LLaMa-3.2's 3B-Instruct LLM from HuggingFace. How can I help you? Please note that we are using free usage of HF's Inference Providers. Thus, only very small models (<7B) can be loaded. Moreover, the providers may have an outage, so if the chat throws an error, please note that it is from the inference provider rather than the agent setup."},
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent
        response = generate_response(message)
        write_message('assistant', response)


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)
