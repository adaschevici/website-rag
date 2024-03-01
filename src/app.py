import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader, AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()


def get_vectorstore_from_url(url):
    loader = AsyncChromiumLoader([url])

    documents = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(documents)

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(docs_transformed)

    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store, document_chunks

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the users's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    return response.get("answer")

st.set_page_config(page_title="Chat with websites", page_icon=":robot:")

st.title("Chat with websites")




with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter a website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL to chat with.")
else:
    # session state initialization
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage("Hello! How can I help you today?"),
    ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)[0]
    # create conversational chain
    vector_store, document_chunks = get_vectorstore_from_url(website_url)
    # st.write(document_chunks)
    user_query = st.chat_input("Type a message...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        #        retrieved_documents = retriever_chain.invoke({
        #            "chat_history": st.session_state.chat_history,
        #            "input": user_query
        #        })
        #        st.write(retrieved_documents)

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
