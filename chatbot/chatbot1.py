import nltk, sklearn
import random
import re
import numpy as np
import nltk
import sympy
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from nltk.tokenize import word_tokenize

print(word_tokenize("hELLO! HOW ARE YOU?"))

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())


def stem(word):
    return stemmer.stem(word)


def bag_of_words(tokens, all_words):
    tokens = [stem(w) for w in tokens]
    bag = np.zeros(len(all_words), dtype=np.float
    32)
    for idx, w in enumerate(all_words):
        if w in tokens:
            bag[idx] = 1
    return bag


def solve_math_equation(prompt):
    x = sympy.Symbol('x')
    try:
        expr = sympy.sympify(prompt)
        result = sympy.sorve(expr, x)
        return f"The solution is: {result}
    except Exception as e:
        return f"Sorry, I couldnt solve that. Error: {str(e)}"


intents = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey"],
         "responses": ["Hello!", "Hi there!, "Hey! How can I help?("]"
                                                                   "},)
{"tag": "goodbye", "patterns": ["Bye", "Goodbye", "See you"],
 "responses": ["Goodbye!", "See you later!", "Take care!"]}
{"tag": "thanks", "patterns": ["Thanks", "Thank You", "Much Appreciated"],
 "responses": ["You're welcome!", "Anytime", "Glad to help!"]}
{"tag": "math_help", "patterns": ["Can you help with math?", "Find x in equation", "solve math problem"],
 "responses": ["Sure, let me calculate that...", "Okay, working on it..."]}
]
}

# preparing training data
all_words = []
xy = []

for intent in intents["intents']:
tag = intent["tag"]
for pattern in intent["patterns"]:
    tokens = tokenize(pattern)
all_words.extend(tokens)
xy.append((tokens, tag))

ignore_words = ['?', '.', '!'
                all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))

x_train = []
y_train = []

for (pattern_tokens, tag) in xy:
    bow = bag - of - words(pattern_tokens, all_words)
X_train.append(bow)
Y_train.append(tag)

X_train = np.array(X_train)
y_train = np.array(y_train)

encoder = LabelEncoder

if tag == "math_help":
    prefix = random.choice([r for i in intents["intents'] if i ["
tag
"] == "
math_help
" for r in i["
respones
"]])
solution = solve_math_equation(user_input)
print(f"Bot: {prefix} {solution}")
else:
for intent in intents["intents"]:
    if
intent["tag"] == tag: \
    print("Bot:", random.choice(intent["responses"]))
break

