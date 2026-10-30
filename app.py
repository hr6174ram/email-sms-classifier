
import streamlit as st
import os
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# 1. Streamlit Page Configuration

st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# 2. Download NLTK Resources

@st.cache_resource
def load_nltk_resources():

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    return set(stopwords.words("english"))


try:
    stop_words = load_nltk_resources()
    ps = PorterStemmer()

except Exception as e:
    st.error(f"NLTK resource error: {e}")
    st.stop()



# 3. Text Preprocessing Function


def transform_text(text):

    text = text.lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Keep only alphanumeric words
    tokens = [word for word in tokens if word.isalnum()]

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)



# 4. Load Trained Model and Vectorizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")


@st.cache_resource
def load_model():

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    with open(vectorizer_path, "rb") as file:
        tfidf = pickle.load(file)

    return model, tfidf


try:
    model, tfidf = load_model()

except FileNotFoundError:
    st.error(
        "Model files not found. "
        "Please check model.pkl and vectorizer.pkl."
    )
    st.stop()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()



# 5. Model Validation


if not hasattr(model, "classes_"):

    st.error("The model is not trained.")
    st.stop()


if not hasattr(tfidf, "vocabulary_"):

    st.error("The TF-IDF vectorizer is not fitted.")
    st.stop()


if model.n_features_in_ != len(tfidf.vocabulary_):

    st.error("Model and vectorizer feature mismatch!")
    st.stop()



# 6. Streamlit User Interface


st.title("📩 Email/SMS Spam Classifier")

st.write(
    "Enter an SMS or Email message below "
    "to check whether it is Spam or Not Spam."
)

st.divider()


# Message input
input_sms = st.text_area(
    "Enter your message",
    placeholder="Type your SMS or Email message here...",
    height=150
)



# 7. Prediction


if st.button("Predict", type="primary", use_container_width=True):

    if not input_sms.strip():

        st.warning("Please enter a message.")

    else:

        try:

            # Text preprocessing
            transformed_sms = transform_text(input_sms)

            if not transformed_sms.strip():

                st.warning(
                    "Please enter a meaningful message "
                    "containing words."
                )

            else:

                # Convert text into numerical features
                vector_input = tfidf.transform([transformed_sms])

                # Model prediction
                result = model.predict(vector_input)[0]

                st.divider()

                # Display result
                if result == 1:

                    st.error("Spam Message Detected!")

                    st.write(
                        "This message is classified as Spam."
                    )

                else:

                    st.success("Not Spam Message")

                    st.write(
                        "This message is classified as legitimate."
                    )

        except Exception as e:

            st.error(f"Prediction error: {e}")



# 8. Footer


st.divider()

st.caption(
    "Developed using Python, NLTK, "
    "TF-IDF and Multinomial Naive Bayes."
)
