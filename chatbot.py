import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import os


os.environ["GOOGLE_API_KEY"] = "YOUR-GOOGLE-API"  
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
    You are a highly knowledgeable AI tutor specialized in Data Science.
    Answer only data science-related queries. If a question is outside this domain, politely refuse.
    
    Conversation history:
    {chat_history}
    
    User: {question}
    AI Tutor:
    """
)

# Initialize the AI model
llm = GoogleGenerativeAI(model="gemini-1.5-pro")
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Streamlit UI setup
st.set_page_config(page_title="Intelligent Data Science Tutor", layout="wide")
st.title("🤖 Intelligent AI Data Science Tutor")

# Initialize session state for messages if not present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a data science question...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get response from AI model
    response = llm_chain.run(question=user_input)
    
    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Add a button to clear conversation
if st.button("Clear Conversation"):
    st.session_state["messages"] = []
    memory.clear()  # Clear conversation memory in LangChain
    st.rerun()
