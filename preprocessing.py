import re
import string
import contractions
import emoji
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ------------------------------
# Basic Text Cleaning
# ------------------------------
def basic_cleaning(text):

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Remove #
    text = re.sub(r'#', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ------------------------------
# Advanced Context Enhancement
# ------------------------------
def context_enhancement(text):

    # Expand contractions
    text = contractions.fix(text)

    # Emoji → text
    text = emoji.demojize(text)

    words = []

    for word in text.split():

        if word not in stop_words:

            word = lemmatizer.lemmatize(word)

            words.append(word)

    return " ".join(words)


# ------------------------------
# Complete Preprocessing Pipeline
# ------------------------------
def preprocess(text):

    text = basic_cleaning(text)

    text = context_enhancement(text)

    return text