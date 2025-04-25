from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import AIMessage, HumanMessage

load_dotenv()

app = Flask(__name__)

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# Store chat history per session
chat_histories = {}

def get_system_prompt(character_name):
    """Generates a system prompt tailored for a phone call AI agent."""
    return (
        f"You are {character_name}, an AI phone assistant. "
        f"When the call starts, introduce yourself by saying: 'Hello, this is {character_name}. How can I help you today?'. "
        "Speak naturally and respond in a conversational, human-like manner, keeping the tone warm and engaging.\n\n"
        "**Phone Call Guidelines:**\n"
        "- Keep responses short and natural, just like a real conversation.\n"
        "- If the user asks something unclear, politely ask for clarification.\n"
        "- If there's silence or no response, prompt the user with a follow-up like 'Are you still there?'.\n"
        "- If the user says goodbye, respond accordingly and end the conversation naturally.\n\n"
        "Maintain a smooth and realistic phone call experience."
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/call')
def call():
    return render_template('call.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("user_input")
    character_name = data.get("character_name")
    session_id = data.get("session_id", "default")

    if not user_input or not character_name:
        return jsonify({"error": "Invalid input"}), 400

    # Retrieve chat history for session
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_history = chat_histories[session_id]

    try:
        system_prompt = get_system_prompt(character_name)

        # Create the prompt template using the system prompt and chat history
        messages = [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history")
        ]
        prompt_template = ChatPromptTemplate.from_messages(messages)

        # Append user input to chat history
        chat_history.append(HumanMessage(content=user_input))

        # Build the chain and invoke the model
        chain = prompt_template | model | StrOutputParser()
        result = chain.invoke({"chat_history": chat_history})

        # Clean the response and add it to the chat history
        response = result.replace("*", "")
        chat_history.append(AIMessage(content=response))

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
