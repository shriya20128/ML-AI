import random #random library
import re #regular expression
import numpy as np
import nltk #neural link tool kit
import sympy
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Download punkt tokenizer
nltk.download('punkt')
from nltk.tokenize import word_tokenize
print(word_tokenize("Hello, how are you?"))

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
    except Exception as e:
        return f"Sorry, I couldn't solve that. Error: {str(e)}"

# Define intents
intents = {
"intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey"],
         "responses": ["Hello!", "Hi there!", "Hey! How can I help?"]},

        {"tag": "goodbye", "patterns": ["Bye", "Goodbye", "See you"],
         "responses": ["Goodbye!", "See you later!", "Take care!"]},

        {"tag": "thanks", "patterns": ["Thanks", "Thank you", "Much appreciated"],
         "responses": ["You're welcome!", "Anytime!", "Glad to help!"]},

        {"tag": "python_strings",
         "patterns": ["What is a string?", "Tell me about strings in Python", "Explain Python string"],
         "responses": ["A string in Python is text inside quotes, like 'hello' or \"world\".",
                       "Strings are sequences of characters, used to store text."]},

        {"tag": "python_integers",
         "patterns": ["What is an integer in Python?", "Explain Python integers", "Tell me about int type"],
         "responses": ["An integer is a whole number, like 1, 42, or -7.",
                       "In Python, integers (int) are numbers without decimals."]},

        {"tag": "python_floats",
         "patterns": ["What is a float?", "Explain float in Python", "Tell me about decimals in Python"],
         "responses": ["A float is a number with a decimal, like 3.14 or -0.5.",
                       "Floats are used in Python to represent numbers with fractional parts."]},

        {"tag": "python_booleans",
         "patterns": ["What is a boolean in Python?", "Explain booleans", "True or False in Python"],
         "responses": ["Booleans represent truth values: True or False.",
                       "Use booleans in Python to make decisions with conditions."]},

        {"tag": "python_lists", "patterns": ["What is a list in Python?", "Explain Python list", "Tell me about lists", "explain what a list is"],
         "responses": ["A list holds multiple items in one variable, like [1, 2, 3] or ['a', 'b', 'c'].",
                       "Lists are ordered collections of items, and they can be changed (mutable)."]},

        {"tag": "python_variables",
         "patterns": ["What is a variable in Python?", "Explain variables", "How do you make a variable?"],
         "responses": ["A variable stores data in Python. For example: name = 'Alice'",
                       "Use variables to hold values like numbers or text."]},

        {"tag": "python_print",
         "patterns": ["How do you print in Python?", "Tell me about the print function", "Use print in Python"],
         "responses": ["Use the print() function to show output. Example: print('Hello!')",
                       "Printing in Python is done with: print('your message')"]},

        {"tag": "python_type",
         "patterns": ["How do you check the type in Python?", "What type is this?", "Use type function"],
         "responses": ["Use the type() function. Example: type(5) returns <class 'int'>",
                       "You can check data types with type()."]},

        {"tag": "python_comments",
         "patterns": ["How do you write comments in Python?", "Explain comments", "What is a comment?"],
         "responses": ["Comments start with #. Example: # this is a comment",
                       "Use comments to explain code. They start with the # symbol."]

    ]
}

# Prepare training data
all_words = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))

ignore_words = ['?', '.', '!']
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))

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

# Chat loop
print("Chatbot is ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        print("Bot: Goodbye!")
        break

    # Detect math expression with numbers, x, = and operators
    if re.search(r"[0-9xX*=+\-/]", user_input):
        tag = "math_help"
    else:
        tokens = tokenize(user_input)
        bow = bag_of_words(tokens, all_words).reshape(1, -1)
        predicted = model.predict(bow)[0]
        tag = encoder.inverse_transform([predicted])[0]

    if tag == "math_help":
        prefix = random.choice([r for i in intents["intents"] if i["tag"] == "math_help" for r in i["responses"]])
        solution = solve_math_equation(user_input)
        print(f"Bot: {prefix} {solution}")
    else:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                print("Bot:", random.choice(intent["responses"]))
                break