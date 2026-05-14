import streamlit as st
from ML import Model
from DP import clean, ents
import spacy

st.set_page_config(
    page_title="Spade - Spam Detection",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def create_model():
    return Model()

st.title("🛡️ Spade")
st.write("Spam Detection using Machine Learning & NLP")

text = st.text_area(
    "Enter email/text",
    height=300,
    placeholder="Paste email text here..."
)

file = st.file_uploader("Upload .txt file", type=["txt"])

email = None
if text.strip():
    email = text
elif file:
    email = file.read().decode("utf-8")

if st.button("🔍 Detect"):
    if not email:
        st.error("Please provide input text.")
    else:
        with st.spinner("Analyzing..."):
            model = create_model()
            cleaned_text, nouns = clean(email)

            vector = model.get_vector(cleaned_text)
            prediction = model.get_prediction(vector)
            probs = model.get_probabilities(vector)

        st.header(f"Prediction: **{prediction}**")

        st.subheader("Model Confidence")
        model_names = [
            "Naive Bayes",
            "Logistic Regression",
            "Random Forest",
            "KNN",
            "SVM"
        ]

        for name, prob in zip(model_names, probs):
            value = int(max(prob) * 100)
            st.write(name)
            st.progress(value)

        st.subheader("📌 Extracted Nouns")
        st.write(", ".join(set(nouns)) if nouns else "No nouns found")

st.header("📌 Named Entity Recognition")

entities = ents(email if email else "")

if entities == "no":
    st.write("No named entities found.")
else:
    for label, values in entities.items():
        with st.expander(label):
            st.caption(spacy.explain(label))
            st.write(", ".join(set(values)))
