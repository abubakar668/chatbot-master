"""
Flask app: serves the business landing page and exposes /chat for the widget.
Run from project root: python app.py
Open http://127.0.0.1:8080 (or set PORT=... to use another port).
"""
import os
import sys

# Run from project root so paths are correct
_project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_project_dir)

from flask import Flask, request, jsonify, send_from_directory

try:
    from chatbot import get_reply
except Exception as e:
    print("Failed to load chatbot:", e, file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

app = Flask(__name__, static_folder="web", static_url_path="")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    reply, end_session = get_reply(message)
    return jsonify({"reply": reply, "end_session": end_session})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server at http://127.0.0.1:{port}")
    app.run(debug=True, port=port)
