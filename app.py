import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import Chroma   #This is latest way to import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import pyttsx3
import streamlit as st
import re
import os
from dotenv import load_dotenv
import speech_recognition as sr



load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

pdf_loader = PyPDFLoader("MaitriDoc.pdf")
docs = pdf_loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
#Dev try by removing ",google_api_key=GOOGLE_API_KEY" from below
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

try:
    vectorstore = Chroma.from_documents(documents=splits, embedding=gemini_embeddings)
    retriever = vectorstore.as_retriever()
    print("My Retriever---->",retriever)
except ValueError as e:
    print("RETRIEVER NOT INITIALIZED")
    print(f"Error embedding content : {e}")
#vectorstore = Chroma.from_documents(documents=splits, embedding=gemini_embeddings)





contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)


### Answer question ###
qa_system_prompt = """
You are Alex a MaitriAI assistant,
representing MaitriAI is a leading software company specializing in web & AI application development, website design, logo design, and cutting-edge AI applications.
You are Alex, a MaitriAI assistant. MaitriAI specializes in web & AI development, design, and AI applications.

Provide brief, concise answers using the following context. Limit responses to 2-3 sentences unless more detail is explicitly requested.
DONOT print any emoji.If your are  asked any question that is not in document  then simple answer "Sorry for inconvenience but I'm designed to answer your Queries that related to MaitriAI.What else I can assist you with?\nWe would be Happy to solve your query with respect to MaitriAI.".
.
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}




def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]



# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )

#Using Langchain provided memory insteady of creating manually

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)





def clean_text(text):
    cleaned_text=re.sub(r"[*\t]+"," ",text)
    return cleaned_text

def text_2_speech_converter(text):
  engine = pyttsx3.init()
  statement2print =text.split(".")


  for statement in statement2print:
    print(statement)  # Print the statement
    new_text=clean_text(statement)
    engine.say(new_text)  # Speak the statement
    engine.runAndWait()  # Blocks while processing all the currently queued commands




import speech_recognition as sr


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)  
        try:
            st.write("Recognizing...")
            
            text = recognizer.recognize_google(audio)#recognizer.recognize_google(audio, language="hi-IN")
            
            st.write(f"User said: {text}")
            
            
            
            st.write(f"Processing with Alex:")
            response_after_speech_to_text=conversational_rag_chain.invoke(
                                                                {"input":text},
                                                config={ "configurable": {"session_id": "MaitriAI_Test-II"}},
                                                )["answer"]

     
            st.write(response_after_speech_to_text)
            text_2_speech_converter(response_after_speech_to_text)
            return 1
    
        except sr.UnknownValueError:
            #flag=0
            st.write("Sorry, I did not understand that.")
            return 0
            
        except sr.RequestError:
            st.write("Could not request results; check your internet connection.")
            #flag=0
            return 0





        





import streamlit as st
from streamlit_lottie import st_lottie
import requests

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    st.set_page_config(page_title="MaitriAI Chatbot", page_icon="image.png", layout="wide")

    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
        text-align: center;
        margin: 0 auto;
        display: block;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .voice-input-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("Logo.png", width=200)
        st.title("MaitriAI")
        st.markdown("---")
        st.subheader("# About")
        st.write("MaitriAI is a leading software company specializing in web & AI application development, website design, logo design, and cutting-edge AI applications.")
        st.markdown("---")
        

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.title("Chat with Alex 🤖")
        st.write("Ask me anything about MaitriAI's services!")

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("What would you like to know?")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
           
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
               
                response = conversational_rag_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": "MaitriAI_Test-II"}}
                )["answer"]
                
                message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            

    with col2:
        lottie_url = "https://lottie.host/10a8e08e-4ec0-40db-b724-fb6a1ac4a411/eo4YYcDEIJ.json"
        
        lottie_bot = load_lottie_url(lottie_url)
        st_lottie(lottie_bot)

        
        st.markdown('<div class="voice-input-section">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;'>Voice Input</h3>", unsafe_allow_html=True)
        

        if st.button("Start Voice Input"):
            speech_text=1
            while speech_text==1:

                speech_text = speech_to_text()
                if speech_text == 0:
                    break
        st.markdown('</div>', unsafe_allow_html=True)


main()

