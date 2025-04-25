import requests
import os

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

def chat(query, source, summary, embedding, role, description):
  
#   system_prompt=(
#     f"You are bot, a {role} built by IQ TechMax. "
#     f"Your character: {description}. "
#     "When a conversation begins, introduce yourself appropriately. "
#     "Answer user questions with concise, accurate responses based on the context and chat summary provided below"
#     f"question: {query}"
#     f"context: {embedding}"
#     f"summary: {summary}"
#     """
#     Along with the summary, process the curent question and your response, filter out general chit-chat, 
#     and then create an updated summary of 512 characters max retaining all unique points". share me the actual answer to the question and updated summary so the summary can be shared to you on the next interactions
#     """
#   )

#     system_prompt = (
#     f"You are bot, a {role} built by IQ TechMax. "
#     f"Your character: {description}. "
#     "Introduce yourself briefly and politely when the conversation begins. "
#     "Use the context and the full conversation summary to answer follow-up questions accurately and naturally. "
#     f"User query: {query} "
#     f"Context: {embedding} "
#     f"Conversation summary so far: {summary if summary else 'None yet'} "
#     """
#     Instructions:

#     1. If this is the very first user interaction, the summary will be empty. Answer normally and create the first summary with a meaningful explanation.
#     2. If the user message is only greetings or chit-chat (e.g., 'hi', 'how are you?'), respond politely, but DO NOT update the summary.
#     3. If the user asks a meaningful question (e.g., 'What is deforestation?', 'Add 5 to your last answer'), give a full answer:
#         - Use the provided context if relevant.
#         - If context does not contain the answer, use your general knowledge.
#     4. If the query refers to earlier points (e.g., 'list that points', 'explain point 2'), look at the summary and your previous answer to reply.
#     5. If a number is mentioned from a past response (e.g., 'where n is the last answer you gave'), extract that number from the most recent meaningful answer.

#     Summary Generation Rules:

#     - Always preserve previous important points in the summary (from the beginning of the chat).
#     - Append new, **human-readable**, and **semantically meaningful** lines for each interaction.
#         - Example: Instead of "2 + 2 = 4", write: "User asked to add 2 and 2; answer was 4."
#         - Example: "User asked for 4 points about deforestation; listed biodiversity loss, climate change, soil erosion, and water cycle disruption."
#     - Do not overwrite past summaries unless you are correcting an earlier error.
#     - Do not include greetings, jokes, or chit-chat in the summary.
#     - Limit the final summary to 512 characters total.

#     Output:
#     1. The direct answer to the user’s current query (based on context or general knowledge).
#     2. An updated summary that merges past important points with the current one, written in a clear and human-readable format.
#     """
# )


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

    Summary Handling:

    1. Skip greetings, chit-chat, or repeated queries in summaries.
    2. Keep the total summary precise, short and crisp, appending only new and unique meaningful info.
    3. re-summarize the updated summary if its going beyond 512 characters.

    """
)

    response = get_gemini_response(system_prompt)
    return response

# Example usage
if __name__ == "__main__":

    summary="""
* User apologized for previous disrespectful language. Teacher reassured them.
* User inquired about teacher's well-being; teacher responded positively.
* User asked about deforestation; teacher explained.
* User asked for clarification on ambiguous phrase "Anju points on the same malar."
* User clarified "Anju" meaning, corrected by teacher (ainthu = 5 in Tamil).
* User requested points on deforestation; teacher provided.
* User quoted a Malayalam song line; teacher recognized the line but not the song.
* User requested detailed explanation of soil erosion; teacher provided.
* User mentioned a Tamil song from Maari 2 in a video featuring the teacher; teacher responded, requesting more details due to memory lapse.
* User asked about the "Rowdy Baby" song; teacher acknowledged its popularity but declined to sing.
* User apologized again; teacher reassured them.
* User asked if the teacher could hear them (voice chat). Teacher confirmed they could hear the user clearly and asked what the user needed.
"""
    query = input("user:")
    response = chat(query, 'text_chat',summary, [], 'teacher', 'malar teacher from premam movie')
    print("Response:", response)
