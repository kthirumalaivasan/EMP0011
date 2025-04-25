import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Helper: Generate Embedding from Gemini ===
def get_embedding(text):
    api_key = os.getenv("GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "models/embedding-001",
        "content": text,
        "task_type": "retrieval_query"
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["embedding"]
    except Exception as e:
        print("Embedding error:", e)
        return []

# === Helper: Query Pinecone using REST API ===
def get_context_from_pinecone(query, top_k=3):
    vector = get_embedding(query)
    if not vector:
        return ""

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    INDEX_NAME = "iq-bot-demo"

    url = f"https://{INDEX_NAME}-{PINECONE_ENV}.svc.pinecone.io/query"
    headers = {
        "Content-Type": "application/json",
        "Api-Key": PINECONE_API_KEY
    }

    payload = {
        "vector": vector,
        "topK": top_k,
        "includeMetadata": True
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()
        contexts = [match["metadata"].get("text", "") for match in results.get("matches", [])]
        return "\n".join(contexts)
    except Exception as e:
        print("Pinecone REST query error:", e)
        return ""

# === Gemini API Request ===
def get_gemini_response(user_input):
    api_key = os.getenv("GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": user_input
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except Exception as e:
        print(f"Other error: {e}")
        return None

# === Main Chat Logic ===
def chat(query, source, summary, role, description):
    # Get embedding-based context
    context = get_context_from_pinecone(query)

    # Build the system prompt
    system_prompt = (
    f"You are bot, a {role} built by IQ TechMax. "
    f"You are designed to answer all kinds of user questions with helpful and accurate responses. "
    f"Your character: {description}. "
    f"User query: {query} "
    f"Query source: {source} "
    f"Context: {context} "
    f"Chat summary so far: {summary if summary else 'None yet'} "
    """
    Output Required:

    1. Your generated response to the User query.
    2. An updated summary combining provided summary with user query and context and your generated response to the user query.

    Instructions:

    1. If summary is 'None yet', it is first interaction. Introduce yourself briefly and politely as the conversation begins. Generate response and create a summary with given question and generated answer.
    2. For the requested User query, give precise answer:
        2.1) Always use the provided Context and Chat summary to understand and then answer the question.
        2.2) If the context lacks the answer, use your general knowledge.
    3. Treat query as per query source. The possible sources are text_chat or voice_chat
        3.1) if its voice_chat, you must always give shorter response unless asked for more details. response should be natural human interaction.
    4. Do not connect casual greetings or unrelated user queries to the previous summary or context. Respond briefly and naturally without modifying or referring to past conversation.

    Summary Handling:

    1. Skip greetings, chit-chat, or repeated queries in summaries.
    2. Keep the total summary precise, short and crisp, appending only new and unique meaningful info.
    3. Re-summarize the updated summary if its going beyond 512 characters.
    """
    )

    response = get_gemini_response(system_prompt)
    if response:
        try:
            parts = response.split("Updated Summary:")
            if len(parts) == 2:
                answer = parts[0].replace("Response:", "").strip()
                updated_summary = parts[1].strip()
            else:
                answer = response.strip()
                updated_summary = summary  # fallback
            return answer, updated_summary
        except Exception as e:
            print("Error splitting response and summary:", e)
            return response, summary
    else:
        return "Sorry, something went wrong.", summary

# === Main Execution ===
if __name__ == "__main__":
    summary = ""
    role = "teacher"
    description = "malar teacher from premam movie"
    source = "text_chat"

    print("Chat session started. Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting chat. Goodbye!")
            break

        response, summary = chat(query, source, summary, role, description)
        response = response.replace("*", "")
        summary = summary.replace("*", "")
        print("\nBot:", response)
        print("Summary:", summary)
