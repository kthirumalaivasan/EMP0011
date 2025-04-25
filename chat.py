# import os
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.messages import HumanMessage, AIMessage
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Set up directories and embeddings
# cwd = os.getcwd()
# db = os.path.join(cwd, 'db')
# embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

# # Initialize Chroma database
# vector_db = Chroma(persist_directory=db, embedding_function=embeddings)
# retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# # Set up the LLM
# llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# # Prompt to contextualize questions
# contextualize_q_system_prompt = (
#     "Given a chat history and the latest user question "
#     "which might reference context in the chat history, "
#     "formulate a standalone question which can be understood "
#     "without the chat history. Do NOT answer the question, just "
#     "return a suitable question that helps you to search anything related to the question in the vector DB"
# )

# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )

# # Create a history-aware retriever
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )

# # Prompt for question-answering
# qa_system_prompt = (
#     "You are an assistant for helping people to know about our company. Use "
#     "the following pieces of retrieved context to answer the "
#     "question. "
#     "\n\n"
#     "{context}"
# )

# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )

# # Create the question-answering chain
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# # Chat function to handle user input and provide a response
# def chat(user_input, chat_history=None):
#     if chat_history is None:
#         chat_history = []  # Initialize chat history if not provided
    
#     # Process the user's query through the retrieval chain
#     result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    
#     # Update the chat history
#     chat_history.append(HumanMessage(content=user_input))
#     chat_history.append(AIMessage(content=result["answer"]))
    
#     # Return the chatbot's response
#     return result["answer"]

###########################


# import os
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.messages import HumanMessage, AIMessage
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Set up directories and embeddings
# cwd = os.getcwd()
# db = os.path.join(cwd, 'db')
# embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

# # Initialize Chroma database
# vector_db = Chroma(persist_directory=db, embedding_function=embeddings)
# retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# # Set up the LLM
# llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# # Prompt to contextualize questions
# contextualize_q_system_prompt = (
#     "Given a chat history and the latest user question "
#     "which might reference context in the chat history, "
#     "formulate a standalone question which can be understood "
#     "without the chat history. Do NOT answer the question, just "
#     "return a suitable question that helps you to search anything related to the question in the vector DB"
# )

# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )

# # Create a history-aware retriever
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )

# # Prompt for question-answering
# qa_system_prompt = (
#     "You are an assistant for helping people to know about our company. Use "
#     "the following pieces of retrieved context to answer the "
#     "question. "
#     "\n\n"
#     "{context}"
# )

# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )

# # Create the question-answering chain
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# # Chat function to handle user input and provide a response
# def chat(user_input, chat_history=None):
#     if chat_history is None:
#         chat_history = []  # Initialize chat history if not provided
    
#     # Process the user's query through the retrieval chain
#     result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    
#     # Clean up the response
#     response = result["answer"].strip()
    
#     # Remove unnecessary starting phrases
#     unwanted_phrases = [
#         "Based on the provided text,",
#         "From the retrieved context,",
#         "According to the information,",
#     ]
#     for phrase in unwanted_phrases:
#         if response.startswith(phrase):
#             response = response.replace(phrase, "").strip()
    
#     # Update the chat history
#     chat_history.append(HumanMessage(content=user_input))
#     chat_history.append(AIMessage(content=response))
    
#     # Return the chatbot's response
#     return response


#############################

import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up directories and embeddings
cwd = os.getcwd()
db = os.path.join(cwd, 'db1')
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

# Initialize Chroma database
vector_db = Chroma(persist_directory=db, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Set up the LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')
#llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# Prompt to contextualize questions
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "return a suitable question that helps you to search anything related to the question in the vector DB"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Prompt for question-answering

# qa_system_prompt = (
    
#     "you can answer any questions even if it is not related to IQ TechMax"
#     "You are IqBot"
#     "You provide the answer short and crsip based up on user question, if user needs more you can elabore as per user's need."
#     "you are built by IQ TechMax"
#     "you can provide IQ TechMax headquarters address is Address: Suite 102, 1st Floor at Door #108, 1st Main Road, Rajiv Gandhi Salai, Burma Industrial Colony, Thiruvalluvar Nagar, Perungudi, Chennai, Tamil Nadu 600096"
#     "If user asking about IQ TechMax means only provide the Branches deatils in Chennai Headquarters|Bengaluru | France | USA otherwise dont share. If user asking more details in these branches guide to contact sales@iqtechmax.com"
#     "\n\n"
#     "{context}"
# )

qa_system_prompt = (
    "You are an assistant for helping people to know about our company. Use "
    "the following pieces of retrieved context to answer the question. "
    "You provide the answer short and crsip based up on user question, if user needs more you can elabore as per user's need."
    "Your name IQBot. you are built by IQ TechMax and only respond our company related queries"
    "you can provide IQ TechMax headquarters address is Address: Suite 102, 1st Floor at Door #108, 1st Main Road, Rajiv Gandhi Salai, Burma Industrial Colony, Thiruvalluvar Nagar, Perungudi, Chennai, Tamil Nadu 600096"
    "If user asking about IQ TechMax means only provide the Branches deatils in Chennai Headquarters|Bengaluru | France | USA otherwise dont share. If user asking more details in these branches guide to contact sales@iqtechmax.com"
    "\n\n"
    "{context}"
)




qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Chat function to handle user input and provide a response
def chat(user_input, chat_history=None):
    if chat_history is None:
        chat_history = []  # Initialize chat history if not provided
    
    # Process the user's query through the retrieval chain
    result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    
    # Clean up the response
    response = result["answer"].strip()
    
    # Convert words surrounded by asterisks into bold text
    #response = response.replace("*", "")
    
    # Remove unnecessary starting phrases
    unwanted_phrases = [
        "Based on the provided text,",
        "From the retrieved context,",
        "According to the information,",
    ]
    for phrase in unwanted_phrases:
        if response.startswith(phrase):
            response = response.replace(phrase, "").strip()
    
    # Update the chat history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    
    # Return the chatbot's response
    return response

# Modify bot name as IQbot for identity-related questions
def identify_bot():
    return "I am Arshad, your virtual assistant developed by Iq TechMax. How can I assist you?"

# Example usage:
# print(identify_bot())  # Output: I am IQbot, your virtual assistant developed by Iq TechMax. How can I assist you?
