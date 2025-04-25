import os
import requests
import json
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "iq-bot-demo1"
CHAT_HISTORY_FILE = "chat_history.txt2"
SUMMARY_FILE = "chat_summary.txt2"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# Helper to append full chat history
def append_chat_history(user_query, bot_response):
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"User: {user_query}\nBot: {bot_response}\n\n")

# Helper to load summary from file
def load_summary():
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

# Helper to save summary to file
def save_summary(summary):
    # Keep summary under 512 characters, trim at word boundary
    if len(summary) > 512:
        summary = summary[:500].rsplit(" ", 1)[0] + "..."
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary.strip())

# Clean and merge meaningful parts of the summary
def update_summary(existing_summary, new_part):
    skip_phrases = ["hi", "hello", "how are you", "what's up", "okay", "thanks"]
    lower_new_part = new_part.lower()
    if any(phrase in lower_new_part for phrase in skip_phrases):
        return existing_summary  # Skip casual talk

    if new_part in existing_summary:
        return existing_summary  # Avoid repetition

    combined = existing_summary + " " + new_part if existing_summary else new_part
    combined = combined.strip()
    if len(combined) > 512:
        combined = combined[:500].rsplit(" ", 1)[0] + "..."
    return combined

# Search full chat history for relevant context
def search_chat_history(query):
    if not os.path.exists(CHAT_HISTORY_FILE):
        return ""
    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        history = f.read()
    # Simple keyword matching (can replace with embedding similarity later)
    relevant_lines = []
    for line in history.split("\n"):
        if any(word.lower() in line.lower() for word in query.split()):
            relevant_lines.append(line.strip())
    return "\n".join(relevant_lines[-10:])  # last 10 relevant lines

# Dummy embedding generator (to be replaced with real model)
def get_embedding(query):
    return [0.0] * 768

# Search Pinecone for similar contexts
def query_pinecone(query):
    vector = get_embedding(query)
    result = index.query(vector=vector, top_k=5, include_metadata=True)
    context = []
    if "matches" in result:
        for match in result["matches"]:
            metadata = match.get("metadata", {})
            if "text" in metadata:
                context.append(metadata["text"])
    return "\n".join(context)

# Get Gemini API response
def get_gemini_response(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None

# Chat function
def chat(query, source, name, role, description):
    current_summary = load_summary()
    context = query_pinecone(query)

    # If the query is not answered by the summary or context
    if not context.strip():
        # Try checking full chat history
        history_context = search_chat_history(query)
        if history_context.strip():
            context = history_context
        else:
            context = "No relevant past context found. Please ask the user to clarify."

    # System prompt template
    system_prompt = f"""
You are {name}, a {role} built by IQ TechMax. You are designed to respond to user questions with helpful and accurate answers strictly based on the provided context. Your character: {description}.

User query: {query}
Query source: {source}
Context (from summary, Pinecone, or history): {context}
Chat summary so far: {current_summary if current_summary else "None yet"}

Instructions:
1. If this is the first interaction (summary is \"None yet\"), briefly introduce yourself with {name} (e.g., \"Hi, I'm {name}. How can I assist you today?\") and then answer the query.
2. Always return your output as a valid JSON object with two keys:
   - 'response': the answer to the query.
   - 'updatedSummary': a crisp, meaningful summary of the conversation so far.
3. Your response must be short, clear, and strictly based on the provided context. Do NOT use external knowledge for domain-specific queries.
4. Use general knowledge only for greetings, small talk, or closings, and do NOT include that in summaries.
5. If the query type is 'voice_chat', make your response conversational and natural, unless more detail is requested.
6. For the summary must be it resummarize in every conversation:
   - Append only new, unique, and meaningful information.
   - Skip greetings, repeated queries, and casual chit-chat.
   - Rephrase or condense the summary if it exceeds 512 characters.
   - The final summary must always be concise, cumulative, and within the character limit.
7. If the answer isn't found in the context, respond with: "I don't have relevant information on that topic based on the current context."
8. If the query involves products or services, attempt to gather the following user details: Name, Contact Number, Email, and Specific Need.

Now, based only on the context provided, answer the following query: {query}
"""

    # Get Gemini response
    response_text = get_gemini_response(system_prompt)

    if response_text and "response" in response_text and "updatedSummary" in response_text:
        try:
            cleaned_response = response_text.replace("```json", "").replace("```", "").strip()
            response_json = json.loads(cleaned_response)

            bot_answer = response_json.get("response", "").strip()
            updated_summary = response_json.get("updatedSummary", "").strip()

            # Directly save the re-summarized content from Gemini
            save_summary(updated_summary)

            # Append the conversation to chat history
            append_chat_history(query, bot_answer)
            return bot_answer
        except Exception as e:
            print(f"Parsing Error: {e}")
            append_chat_history(query, response_text)
            return response_text
    else:
        append_chat_history(query, response_text or "No response")
        return response_text or "Sorry, no response from Gemini."

# CLI usage
if __name__ == "__main__":
    print("Start. Type 'exit' to stop.\n")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chat ended.")
            break
        answer = chat(
            query=user_input,
            source="text_chat",
            name="ktm",
            role="AI Assistant",
            description="A helpful and strict assistant with access only to provided context."
        )
        print("Bot:", answer)
