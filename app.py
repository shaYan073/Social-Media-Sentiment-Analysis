# Import required libraries
import streamlit as st
import joblib

from preprocessing import preprocess

# Load the saved model
model = joblib.load("linear_svc_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load Chi-Square feature selector
selector = joblib.load("feature_selector.pkl")

# Title of the web application
st.title("Social Media Sentiment Analysis")

# Short description
st.write("Enter any social media text below to predict its sentiment.")

# Text input box
user_input = st.text_area("Enter Text")

# Predict button
if st.button("Predict Sentiment"):

    # Clean the user text
    clean_text = preprocess(user_input)

    # Convert cleaned text into TF-IDF features
    text_vector = vectorizer.transform([clean_text])

    # Reduce features using Chi-Square selector
    text_vector = selector.transform(text_vector)

    # Predict sentiment
    prediction = model.predict(text_vector)

    # Display prediction
    st.success(f"Predicted Sentiment: {prediction[0]}")