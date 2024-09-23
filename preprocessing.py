import re
import json
import string
import emoji
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Load chat words and number meanings if needed
with open(r'chat_words.json','r') as f:
    chat_words = json.load(f)

with open(r'number_meanings.json','r') as f:
    number_meanings = json.load(f)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    # Lowercase text
    text = text.lower()

    # Clean spaces
    text = ' '.join(text.split())

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Replace chat words
    new_text = []
    for word in text.split():
        if word.upper() in chat_words:
            new_text.append(chat_words[word.upper()])
        else:
            new_text.append(word)
    text = " ".join(new_text)

    # Convert emojis to text
    text = emoji.demojize(text, delimiters=("", ""))

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text
    tokens = word_tokenize(text)

    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]

    # Join tokens back into sentence
    text = ' '.join(tokens)

    return text