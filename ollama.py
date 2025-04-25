# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# import streamlit as st

# st.title("Thiru's AI")
# st.subheader("Anything I can do for you")

# input_txt = st.text_input("What’s on your mind? Type it here!......")

# prompt = ChatPromptTemplate.from_messages(
#     [("system","you are a helpful AI Assistant. your name is ktm the bot"),
#      ("user","user query:{query}")
#      ])

# llm = Ollama(model= "llama2")
# output_parser = StrOutputParser()
# chain = prompt|llm|output_parser

# if input_txt:
#     st.write(chain.invoke({"query":input_txt}))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os

app = FastAPI()

# Ensure the "static" directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Setup templates and static directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=".")

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM setup
llm = Ollama(model="llama2")
prompt = ChatPromptTemplate.from_messages([ 
    ("system", "You are a helpful AI assistant. Your name is KTM the bot."), 
    ("user", "user query:{query}") 
])
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Store chat history in memory (for this demo)
chat_history = []

@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("chat_template.html", {"request": request, "chat_history": chat_history})

@app.post("/send", response_class=HTMLResponse)
async def send_message(request: Request, user_input: str = Form(...)):
    user_msg = user_input.strip()
    if user_msg:
        chat_history.append(("user", user_msg))
        bot_reply = chain.invoke({"query": user_msg})
        chat_history.append(("bot", bot_reply))
    return templates.TemplateResponse("chat_template.html", {"request": request, "chat_history": chat_history})

# Inline template using Jinja2
with open("chat_template.html", "w") as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Thiru's AI Chat</title>
    <style>
        body {
            font-family: "Segoe UI", sans-serif;
            background: #fff;
            display: flex;
            justify-content: center;
            margin: 0;
        }
        .container {
            max-width: 700px;
            width: 100%;
            padding: 30px;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        h1 {
            text-align: center;
            font-size: 2rem;
            margin-bottom: 20px;
        }
        .chat-box {
            flex-grow: 1;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 12px;
            background: #f9f9f9;
        }
        .message {
            padding: 10px 15px;
            border-radius: 12px;
            margin: 10px 0;
            max-width: 80%;
            line-height: 1.4;
        }
        .user {
            background-color: #DCF8C6;
            align-self: flex-end;
            margin-left: auto;
        }
        .bot {
            background-color: #F1F0F0;
            align-self: flex-start;
            margin-right: auto;
        }
        .input-area {
            display: flex;
            margin-top: 20px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ccc;
            font-size: 1rem;
        }
        button {
            margin-left: 10px;
            padding: 12px 20px;
            border: none;
            background-color: #2e7d32;
            color: white;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>What can I help with?</h1>
        <div class="chat-box">
            {% for role, msg in chat_history %}
                <div class="message {{ role }}">{{ msg }}</div>
            {% endfor %}
        </div>
        <form method="post" action="/send" class="input-area">
            <input type="text" name="user_input" placeholder="Ask anything..." required />
            <button type="submit">Send</button>
        </form>
    </div>
</body>
</html>
''')

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

