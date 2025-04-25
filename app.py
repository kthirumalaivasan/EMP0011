from flask import Flask, render_template, request, jsonify
from cchat import chat
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/iqbot", methods=["POST"])
def chat_api():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"response": "Invalid input!"}), 400
    response = chat(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
