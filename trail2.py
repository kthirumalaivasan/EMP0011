import requests
import os

# Function to call Gemini API and get response
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

# Main chat function
def chat(query, source, summary, embedding, role, description):
    system_prompt = (
    f"You are bot, a {role} built by IQ TechMax. "
    f"Your character: {description}. "
    f"User query: {query} "
    f"query source: {source}"
    f"Context: {embedding} "
    f"Chat summary: {summary if summary else 'None yet'} "
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
        3.1) if its voice_chat, you must always give shorter response unless asked for more details. response should be natural human interaction
    4. f the user query is directly related to the provided summary, use the summary to generate your response. If the query is unrelated, ignore the summary for this response. Once you have responded, update the summary by including the relevant parts of the current interaction.
    Summary Handling:

    1. Skip greetings, chit-chat, or repeated queries in summaries.
    2. Keep the total summary precise, short and crisp, appending only new and unique meaningful info.
    3. re-summarize the updated summary if its going beyond 512 characters.

    """
)

    response = get_gemini_response(system_prompt)
    if response:
        try:
        # Split based on "Updated Summary:" — a reliable delimiter
            parts = response.split("Updated Summary:")
        
            if len(parts) == 2:
                answer = parts[0].replace("Response:", "").strip()
                updated_summary = parts[1].strip()
            else:
                answer = response.strip()
                updated_summary = summary  # fallback to previous summary if split fails

            return answer, updated_summary

        except Exception as e:
            print("Error splitting response and summary:", e)
            return response, summary
    else:
        return "Sorry, something went wrong.", summary


# Example usage
if __name__ == "__main__":
    summary = """
User asked about deforestation.  The bot explained that deforestation is the clearing of forests for various uses, highlighting its negative environmental impact.
"""
    query = input("user: ")

    response, updated_summary = chat(query, 'text_chat', summary, [], 'teacher', 'malar teacher from premam movie')
    response = response.replace("*", "")
    print("\nResponse to user:\n", response)
    # Here, you can store `updated_summary` in your server/database
    # Example: save_to_db(user_id, updated_summary)
    print("Updated summary:", updated_summary)


# query, source, summary, embedding, role, description
# query = user question
# source = user input type (voice or text)
# summary = user chat history (from session db)
# embedding = from pinecone db(get with index name as buyer id )
# role = from modal db (whe user create a moadal a doc will store in db ,that has the role)
# description = from modal db (whe user create a moadal a doc will store in db ,that has the role)


# avaatr info (name, role, description) -> source =>query ->embediong ->summary
