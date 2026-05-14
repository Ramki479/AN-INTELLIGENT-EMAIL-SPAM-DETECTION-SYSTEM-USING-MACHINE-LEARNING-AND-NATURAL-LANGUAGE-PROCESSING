import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
import spacy

# --------- NLTK AUTO DOWNLOAD ---------
required_nltk = [
    "stopwords",
    "punkt",
    "averaged_perceptron_tagger",
    "wordnet",
    "omw-1.4",
]

for pkg in required_nltk:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

# --------- LOAD RESOURCES ---------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

tag_map = defaultdict(lambda: wn.NOUN)
tag_map["J"] = wn.ADJ
tag_map["V"] = wn.VERB
tag_map["R"] = wn.ADV


def process_sentence(sentence):
    base_words = []
    final_words = []
    nouns = []

    sentence = re.sub(r"[^\w\s]", "", sentence)
    sentence = re.sub(r"_", " ", sentence)

    words = word_tokenize(sentence)
    pos_tagged_words = pos_tag(words)

    for token, tag in pos_tagged_words:
        base_words.append(
            lemmatizer.lemmatize(token, tag_map[tag[0]])
        )

    for word in base_words:
        if word not in stop_words:
            final_words.append(word)

    for token, tag in pos_tagged_words:
        if tag == "NN" and len(token) > 1:
            nouns.append(token)

    return " ".join(final_words), nouns


def clean(email: str):
    email = email.lower()
    sentences = sent_tokenize(email)

    total_nouns = []
    cleaned_text = []

    for sent in sentences:
        sentence, nouns = process_sentence(sent)
        cleaned_text.append(sentence)
        total_nouns.extend(nouns)

    return " ".join(cleaned_text), total_nouns


def ents(text: str):
    if not text.strip():
        return "no"

    doc = nlp(text)
    entities = {}

    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)

    return entities if entities else "no"
