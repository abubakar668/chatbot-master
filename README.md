# Chatbot Project

A conversational chatbot that answers questions, handles greetings, and responds to predefined prompts. Built with Python, NLTK, and scikit-learn (TF-IDF and cosine similarity) for intent matching and corpus-based replies.

---

## Features

* **Greeting detection** — Responds to hello, hi, hey, and similar inputs
* **Predefined Q&A** — Covers common questions (e.g. capitals, facts, jokes, recommendations) with flexible matching (punctuation and extra words allowed)
* **Corpus-based answers** — Falls back to similarity matching against a text corpus for open-ended queries
* **Help command** — Users can type `help` or "what can you do" to see example prompts
* **Session summary** — On exit, displays the number of messages in the session

---

## Requirements

* [Python](https://www.python.org/downloads/) 3.x

---

## Installation

1. **Clone or download the project**

   ```bash
   git clone https://github.com/your-username/chatbot-project.git
   cd chatbot-project
   ```

2. **(Recommended) Create a virtual environment**

   ```bash
   python -m venv venv
   ```

   * **Windows:** `venv\Scripts\activate`
   * **macOS/Linux:** `source venv/bin/activate`

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Chatbot

**Terminal (CLI):** From the project directory:

```bash
python chatbot.py
```

Type questions or commands; type `bye` or `thanks` to end the session.

**Web (business page + chat widget):** A demo site with the chatbot in a bottom-left widget:

```bash
python app.py
```

Then open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser. To use another port: `PORT=5050 python app.py`. The page is presentation-only; only the chat widget is functional.

---

## Customization

Predefined responses and behavior can be adjusted in `chatbot.py`:

* **Additional responses** — Edit the `ADDITIONAL_RESPONSES` dictionary to add or change Q&A pairs
* **Corpus** — Replace or edit `chatbot.txt` to change the knowledge base used for similarity-based answers
* **Greetings** — Update `GREETING_INPUTS` and `GREETING_RESPONSES` to change greeting detection and replies

---

## Project Structure

| File          | Purpose                                      |
|---------------|----------------------------------------------|
| `chatbot.py`  | Main chatbot logic, matching, and conversation loop |
| `chatbot.txt` | Text corpus for similarity-based responses   |
| `app.py`      | Flask app: serves the web page and `/chat` API for the widget |
| `web/index.html` | Business landing page and chat widget UI   |
| `requirements.txt` | Python dependencies                    |

---

## Support

For bugs or change requests, open an issue or pull request in the repository.
