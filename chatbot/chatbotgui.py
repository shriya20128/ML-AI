# Fix SSL issues and set custom NLTK data path
import os
import ssl
import certifi
import nltk
import json

# Use certifi certificates for SSL downloads
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context
from nltk.data import find
# Set custom path to your punkt tokenizer
nltk.download('punkt')
print("Punkt path:", find('tokenizers/punkt'))
nltk.data.path.append('/Users/shriy/nltk_data')  # <- Replace with your real path


# Begin chatbot setup
import tkinter as tk
from tkinter import scrolledtext
import random
import numpy as np
import sympy
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Helper functions
stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())
def stem(word):
    return stemmer.stem(word)
def bag_of_words(tokens, all_words):
    tokens = [stem(w) for w in tokens]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokens:
            bag[idx] = 1
    return bag
def solve_math_equation(prompt):
    x = sympy.Symbol('x')
    try:
        expr = sympy.sympify(prompt)
        result = sympy.solve(expr, x)
        return f"The solution is: {result}"
    except:
        return "Sorry, I couldn't understand that. Please writing something related like - want to learn strings'"

# Define intents
# Load the intents.json file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Prepare training data
all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))

ignore_words = ['?', '.', '!']
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(intent["tag"] for intent in intents["intents"]))

X_train = []
y_train = []

for (pattern_tokens, tag) in xy:
    bow = bag_of_words(pattern_tokens, all_words)
    X_train.append(bow)
    y_train.append(tag)

X_train = np.array(X_train)
y_train = np.array(y_train)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
model = MultinomialNB()
model.fit(X_train, y_train)

# GUI
def send_message():
    user_input = entry.get()
    chat_log.insert(tk.END, "You: " + user_input + '\n')
    entry.delete(0, tk.END)

    tokens = tokenize(user_input)
    bow = bag_of_words(tokens, all_words).reshape(1, -1)
    predicted = model.predict(bow)[0]
    tag = encoder.inverse_transform([predicted])[0]

    if tag == "python_strings":
        chat_log.insert(tk.END, "AI Agent: " + random.choice(
            [r for i in intents["intents"] if i["tag"] == "python_strings" for r in i["responses"]]) + '\n')
        chat_log.insert(tk.END, "AI Agent: " + solve_math_equation(user_input) + '\n')
    else:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                chat_log.insert(tk.END, "AI Agent: " + random.choice(intent["responses"]) + '\n')
                break

# Launch GUI
root = tk.Tk()
root.title("AI Agent teaching Python Fundamentals")

chat_log = scrolledtext.ScrolledText(root, width=130, height=36, bg="light blue")
chat_log.pack()

entry = tk.Entry(root, width=50)
entry.pack(pady=10)

send_button = tk.Button(root, text="Send", command=send_message)
# Make Enter key trigger the send_message function
send_button.pack()

root.mainloop()
