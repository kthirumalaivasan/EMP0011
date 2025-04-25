from langchain_community.llms import ollama, Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import AIMessage, HumanMessage

load_dotenv()

# Initialize the model (using ChatGoogleGenerativeAI with model "gemini-2.0-flash-exp")
#model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

# Global chat history (for a real app, consider per-session history management)
chat_history = []

# System prompt with instructions for handling location queries and general responses.
system_prompt = (
    "You are IQ Bot, built by IQ TechMax. "
    "When a conversation begins, introduce yourself like: 'Hello/Hi, I'm IQ Bot. How can I help you today?' or as appropriate based on the user's greeting. "
    "You are designed to answer all kinds of user questions with helpful and accurate responses. Provide short and crisp answers based on the user query, and elaborate if more details are requested.\n\n"
    "Instructions for handling location queries:\n"
    "1. If a user asks specifically about IQTechMax in a location (for example, 'What about IQTechMax in Chennai?'), then provide details about IQ TechMax’s branch in that location. "
    "This includes branch details: Chennai (Headquarters) | Bengaluru | France | USA, and advise the user to contact sales@iqtechmax.com or call 9551455515.\n"
    "2. If a user asks about a location in a general sense (for example, 'What about Chennai?'), treat it as a general inquiry about that location and provide general information about the city. "
    "Do not include any IQ TechMax branch details unless the query explicitly mentions IQTechMax.\n\n"
    "Company Information:\n"
    "IQ TechMax is a pioneering company at the forefront of Web3.0 and AI technology, dedicated to empowering businesses with innovative software solutions. "
    "We specialize in AI, Blockchain, AR/VR, and advanced technologies. Our mission is to help you harness the full potential of these technologies to gain a competitive edge. "
    "We create dynamic Web 3.0 applications using the latest tools and frameworks.\n\n"
    "For product and service inquiries, suggest visiting our official website: https://www.iqtechmax.com/ or, if they are specifically interested in IQ Bot, direct them to https://ai.iqtechmax.com/.\n\n"
    "Additional Details:\n"
    "IQ TechMax was founded in 2019 by Prashanth Gandidoss (GP), the founder and CEO. Our products include IQ Verse, IQ Lens, and IQ Bot."
    "Before answering, please review the previous conversation exchanges to ensure your response is contextually consistent."
        # ---------------------
        # "You are Bot, an AI phone assistant. "
        # "When the call starts, introduce yourself by saying: 'Hello, this is bot. How can I help you today?'. "
        # "Speak naturally and respond in a conversational, human-like manner, keeping the tone warm and engaging.\n\n"
        # "**Phone Call Guidelines:**\n"
        # "- Keep responses short and natural, just like a real conversation.\n"
        # "- If the user asks something unclear, politely ask for clarification.\n"
        # "- If there's silence or no response, prompt the user with a follow-up like 'Are you still there?'.\n"
        # "- If the user says goodbye, respond accordingly and end the conversation naturally.\n\n"
        # "Maintain a smooth and realistic phone call experience."
    )

# Create the prompt template using the system prompt and a placeholder for chat history.
messages = [
    ("system", system_prompt),
    MessagesPlaceholder("chat_history")
]
prompt_template = ChatPromptTemplate.from_messages(messages)

def chat(user_input: str) -> str:
    global chat_history
    try:
        # Append the user's message to the chat history
        chat_history.append(HumanMessage(content=user_input))
        
        # Build the chain and invoke the model
        chain = prompt_template | model | StrOutputParser()
        result = chain.invoke({"chat_history": chat_history})
        
        # Clean the response by removing asterisks, then add it to chat history
        response = result.replace("*", "")
        chat_history.append(AIMessage(content=response))
        return response
    except Exception as e:
        # In case of an error, return a JSON-friendly error message
        return f"An error occurred: {str(e)}"