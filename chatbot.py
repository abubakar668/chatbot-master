#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)

# uncomment the following only the first time
nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only

# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

#Tokenization
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

#Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def normalize_for_match(text):
    """Strip punctuation and collapse spaces for flexible response matching."""
    cleaned = text.lower().translate(remove_punct_dict)
    return ' '.join(cleaned.split())

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """if user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

ADDITIONAL_RESPONSES = {
    "how are you": "I'm just a chatbot, but thanks for asking!",
    "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
    "who created you": "I was created by a team of developers, mostly Henry",
    "bye": "Goodbye! Feel free to come back if you have more questions.",
    "what's your favorite color": "I don't have personal preferences, but I can help with your questions.",
    "who won the World Series in 2020": "The Los Angeles Dodgers won the World Series in 2020.",
    "what's the weather like today": "I don't have access to real-time data, but you can check a weather website or app.",
    "recommend a book": "It depends on your interests. Do you prefer fiction or non-fiction?",
    "what's the meaning of life": "The meaning of life is a profound and philosophical question. It varies from person to person.",
    "how can I learn programming": "You can start by learning a programming language like Python and practice regularly.",
    "what's your favorite movie": "I don't watch movies, but I can discuss movie recommendations.",
    "where is the Eiffel Tower located": "The Eiffel Tower is located in Paris, France.",
    "tell me about artificial intelligence": "Artificial intelligence (AI) is the simulation of human intelligence by machines.",
    "who is your favorite celebrity": "I don't have preferences, but I can provide information about various celebrities.",
    "what's the capital of Japan": "The capital of Japan is Tokyo.",
    "how does a computer work": "A computer processes data using a combination of hardware and software.",
    "do you like pizza": "I can't eat, but I can help you find pizza places near you.",
    "tell me a fun fact": "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible.",
    "what's the largest planet in our solar system": "Jupiter is the largest planet in our solar system.",
    "how does a search engine work": "Search engines use web crawlers to index websites and algorithms to rank and display search results.",
    "tell me a riddle": "I'm in the middle of water but never get wet. What am I? Answer: A shadow.",
    "what's the population of India": "As of my last update in 2021, India's population is over 1.3 billion people.",
    "what's the square root of 144": "The square root of 144 is 12.",
    "recommend a TV show": "What genre are you interested in? Comedy, drama, science fiction, or something else?",
    "what's the difference between HTML and CSS": "HTML is used for structuring web content, while CSS is used for styling and layout.",
    "tell me about famous scientists": "There are many famous scientists, like Albert Einstein, Isaac Newton, and Marie Curie.",
    "how can I stay healthy": "Staying healthy involves a balanced diet, regular exercise, and good sleep habits.",
    "what's the capital of Australia": "The capital of Australia is Canberra.",
    "tell me a famous quote": "Here's one by Albert Einstein: 'Imagination is more important than knowledge.'",
    "how do I create a website": "You can create a website using HTML, CSS, and web development tools like WordPress or Wix.",
    "what's the boiling point of water": "The boiling point of water at sea level is 100 degrees Celsius or 212 degrees Fahrenheit.",
    "tell me a travel tip": "When traveling, pack light and make a list of important items to avoid forgetting anything.",
    "what's the speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second (or about 186,282 miles per second).",
    "tell me about famous authors": "Famous authors include William Shakespeare, Jane Austen, and George Orwell.",
    "how do I improve my time management": "Effective time management involves setting goals, prioritizing tasks, and avoiding procrastination.",
    "what's the longest river in the world": "The Nile River is the longest river in the world.",
    "tell me a historical fact": "In 1969, astronauts Neil Armstrong and Buzz Aldrin became the first humans to walk on the moon.",
    "how do I start a small business": "Starting a small business involves planning, financing, and marketing.",
    "what's the largest desert in the world": "The largest desert in the world is the Antarctic Desert, followed by the Arctic Desert.",
    "tell me a cooking tip": "When cooking, taste your food as you go and adjust seasonings to your preference.",
    "what's the capital of Brazil": "The capital of Brazil is Brasília.",
    "how can I become a better public speaker": "To become a better public speaker, practice and refine your speaking skills and confidence.",
    "tell me a space fact": "Space is completely silent because there is no air to carry sound waves.",
    "what's the smallest prime number": "The smallest prime number is 2.",
    "how do I create a strong password": "Create a strong password by using a combination of letters, numbers, and special characters.",
    "tell me about famous artists": "Famous artists include Leonardo da Vinci, Vincent van Gogh, and Pablo Picasso.",
    "what's the coldest place on Earth": "The coldest place on Earth is Antarctica, where temperatures can drop below -100 degrees Fahrenheit.",
    "tell me a science experiment to try at home": "You can make a volcano using baking soda and vinegar for a fun and safe experiment.",
    "what's the largest ocean in the world": "The Pacific Ocean is the largest ocean in the world.",
    "how can I reduce stress": "Reducing stress involves relaxation techniques, exercise, and managing your workload.",
    "tell me a technology fact": "In 1991, the World Wide Web (WWW) was introduced, changing the way we access information.",
    "what's the capital of China": "The capital of China is Beijing.",
    "tell me a proverbs or saying": "Here's one: 'Actions speak louder than words.'",
    "what's the highest mountain in the world": "Mount Everest is the highest mountain in the world.",
    "how can I start a healthy diet": "Start a healthy diet by incorporating more fruits, vegetables, and whole grains into your meals.",
    "tell me a music fact": "The Beatles' 'Yesterday' is one of the most covered songs in music history.",
    "what's the largest animal in the world": "The blue whale is the largest animal on Earth.",
    "tell me a gardening tip": "Water your plants in the morning to prevent fungal growth and conserve water.",
    "what's the currency of Canada": "The currency of Canada is the Canadian dollar (CAD).",
    
}

# Help command: example prompts to show when user asks for help
HELP_MESSAGE = (
    "I can chat, answer questions, and tell jokes! Try asking me:\n"
    "  • 'Tell me a joke'  • 'How are you?'  • \"What's the capital of Japan?\"\n"
    "  • 'Tell me a fun fact'  • 'Tell me about artificial intelligence'\n"
    "Type 'bye' to exit, or 'thanks' when you're done."
)
HELP_TRIGGERS = {"help", "what can you do", "what can you do for me", "options", "commands"}

def get_additional_response(user_response):
    """Find a matching predefined response: exact match, normalized match, or key contained in input."""
    normalized_input = normalize_for_match(user_response)
    # Exact match (original or normalized)
    if user_response in ADDITIONAL_RESPONSES:
        return ADDITIONAL_RESPONSES[user_response]
    if normalized_input in ADDITIONAL_RESPONSES:
        return ADDITIONAL_RESPONSES[normalized_input]
    # Any key contained in user input (prefer longest match)
    for key in sorted(ADDITIONAL_RESPONSES.keys(), key=len, reverse=True):
        if key in normalized_input or normalize_for_match(key) in normalized_input:
            return ADDITIONAL_RESPONSES[key]
    return None

# Update the response function to include additional responses
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        robo_response = "I am sorry, I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    
    # Check if the user input has an additional response (exact or smart match)
    additional = get_additional_response(user_response)
    if additional is not None:
        robo_response = additional

    return robo_response

# Chatbot conversation loop
flag = True
session_message_count = 0
print("Julie: My name is Julie. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while flag:
    user_response = input("You: ")
    user_response = user_response.lower()

    if user_response != 'bye':
        if user_response in ('thanks', 'thank you'):
            flag = False
            session_message_count += 1
            print("Julie: You're welcome.")
        elif normalize_for_match(user_response) in HELP_TRIGGERS:
            session_message_count += 1
            print("Julie: " + HELP_MESSAGE)
        else:
            if greeting(user_response) is not None:
                session_message_count += 1
                print("Julie: " + greeting(user_response))
            else:
                session_message_count += 1
                print("Julie: " + response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        session_message_count += 1
        print("Julie: Goodbye! Feel free to come back if you have more questions.")

    if not flag:
        print(f"\nSession ended. You had {session_message_count} messages with Julie. See you next time!")
