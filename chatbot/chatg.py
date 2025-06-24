#import library
import nltk
import random
import re
import numpy as np
import sympy
import json
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB


#punkt
nltk.download("punkt")
print("Libraries loaded!")
#Tokenize
from nltk.tokenize import word_tokenize
print(word_tokenize("Hello how are you?"))

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
        result = sympy.solve(expr,x)
        return f"The solution is: {result}"
    except Exception as e:
        return f"Sorry, I couldnt solve that. Error: {str(e)}"

# Load the intents.json file
with open("intents.json", "r") as file:
    intents = json.load(file)

all_words = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens,tag))

ignore_words = ['?','.','!']
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))

X_train = []
Y_train = []

for (pattern_tokens, tag) in xy:
    bow = bag_of_words(pattern_tokens, all_words)
    X_train.append(bow)
    Y_train.append(tag)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)

model = MultinomialNB()
model.fit(X_train,Y_train)

#chat loop
print("Chatbot is ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You:  ")
    if user_input.lower() in ['quit','exit']:
        print("Bot: Goodbye!")
        break

    if re.search(r"[0-9xX*=+\-/]", user_input):
        tag = "math_help"
    else:
        tokens = tokenize(user_input)
        bow = bag_of_words(tokens, all_words).reshape(1,-1)
        predicted = model.predict(bow)[0]
        tag = encoder.inverse_transform([predicted])[0]

    if tag == "math_help":
        prefix = random.choice([r for i in intents["intents"] if i["tag"] == "math_help" for r in i["responses"]])
        solution = solve_math_equation(user_input)
        print(f"Bot: {prefix} {solution}")
    else:
        for intent in intents["intents"]:
            if intent ["tag"] == tag:
                print("Bot:", random.choice (intent["responses"]))
                break
