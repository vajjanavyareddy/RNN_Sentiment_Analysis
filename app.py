
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import string

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Mental Health Sentiment Monitor",
    page_icon="🧠",
    layout="wide"
)


st.markdown(
    """
    <style>

    .main {
        background-color: white;
    }

    .title-box {
        background: linear-gradient(90deg, #6A11CB, #8E2DE2);
        padding: 28px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    }

    .title-box h1 {
        margin: 0;
        font-size: 40px;
        font-weight: 700;
    }

    .title-box p {
        margin-top: 10px;
        font-size: 18px;
        opacity: 0.95;
    }

    .section-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 25px;
        border: 1px solid #f0f0f0;
    }

    .result-box {
        background-color: #F7F3FF;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #8E2DE2;
        margin-top: 15px;
    }

    .tip-box {
        background-color: #F3F8FF;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #4B8BBE;
        margin-top: 15px;
    }

    textarea {
        font-size: 16px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


model = load_model("mental_health_rnn_model.h5")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)



st.markdown(
    """
    <div class="title-box">
        <h1>AI-Based Mental Health Sentiment Monitoring System</h1>
        <p>Emotion Detection using Simple Recurrent Neural Networks</p>
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown('<div class="section-box">', unsafe_allow_html=True)

st.header("📌 About the Project")

st.write(
    """
    This project uses Natural Language Processing (NLP) and a Simple Recurrent Neural Network (RNN)
    to analyze emotional sentiment from user text messages.

    The system helps in:
    - detecting emotional patterns
    - monitoring mental wellness
    - identifying negative sentiment trends
    - supporting emotional awareness

    RNN models are highly effective for sequence learning because they remember previous words
    while processing text, helping the model understand emotional context.
    """
)

st.markdown('</div>', unsafe_allow_html=True)



st.markdown('<div class="section-box">', unsafe_allow_html=True)

st.header("📝 Enter Your Thoughts")

st.write("### Sample Sentences")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("I feel mentally tired and stressed")

with col2:
    st.info("I am feeling peaceful and happy today")

with col3:
    st.info("I do not feel like talking to anyone")

user_input = st.text_area(
    "User Text Input",
    placeholder="Enter your thoughts or feelings here...",
    height=180
)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Emotional Guidance Messages
# -------------------------------------------------

guidance = {
    "Anxiety": {
        "message": "Take a short break and practice deep breathing.",
        "activity": "Try meditation or a short walk outdoors."
    },

    "Stress": {
        "message": "Your mind may need relaxation and rest.",
        "activity": "Listen to calming music or take proper sleep."
    },

    "Depression": {
        "message": "Please stay connected with supportive people.",
        "activity": "Engage in light activities and avoid isolation."
    },

    "Suicidal": {
        "message": "Please seek immediate emotional support.",
        "activity": "Reach out to trusted people or professionals."
    },

    "Normal": {
        "message": "You seem emotionally balanced today.",
        "activity": "Continue maintaining healthy routines and positivity."
    },

    "Bipolar": {
        "message": "Maintaining emotional balance is important.",
        "activity": "Follow healthy routines and stress management habits."
    },

    "Personality disorder": {
        "message": "Self-awareness and communication are valuable.",
        "activity": "Practice mindfulness and positive social interaction."
    }
}

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------

if st.button("🔍 Analyze Emotion", use_container_width=True):

    if user_input.strip() == "":

        st.warning("Please enter some text for analysis.")

    else:

        # -----------------------------
        # Preprocessing
        # -----------------------------

        text = user_input.lower()

        text = text.translate(
            str.maketrans('', '', string.punctuation)
        )

        # -----------------------------
        # Sequence Conversion
        # -----------------------------

        sequence = tokenizer.texts_to_sequences([text])

        max_len = 20

        padded = pad_sequences(
            sequence,
            maxlen=max_len,
            padding='post'
        )

        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = model.predict(padded)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        sentiment = label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        # -------------------------------------------------
        # Prediction Output
        # -------------------------------------------------

        st.markdown('<div class="section-box">', unsafe_allow_html=True)

        st.header("📊 Prediction Result")

        st.markdown(
            f"""
            <div class="result-box">
                <h3>Emotion Detected: {sentiment}</h3>
                <h4>Confidence Score: {confidence * 100:.2f}%</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        if confidence > 0.85:
            st.success("Strong emotional prediction detected.")
        elif confidence > 0.60:
            st.info("Moderate emotional prediction detected.")
        else:
            st.warning("Prediction confidence is low.")

        st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------------------------------
        # Visualization Section
        # -------------------------------------------------

        st.markdown('<div class="section-box">', unsafe_allow_html=True)

        st.header("📈 Sentiment Confidence Graph")

        class_names = label_encoder.classes_

        probabilities = prediction[0]

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.bar(class_names, probabilities)

        ax.set_xlabel("Sentiment Category")
        ax.set_ylabel("Confidence Probability")
        ax.set_title("Emotion Probability Distribution")

        plt.xticks(rotation=15)

        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------------------------------
        # Emotional Guidance Section
        # -------------------------------------------------

        st.markdown('<div class="section-box">', unsafe_allow_html=True)

        st.header("💡 Emotional Wellness Guidance")

        st.markdown(
        f"""
            <div class="tip-box">
            <h3>Motivational Message</h3>
            <p>{guidance[sentiment]['message']}</p>

         <h3>Suggested Positive Activity</h3>
         <p>{guidance[sentiment]['activity']}</p>
         </div>
         """,
         unsafe_allow_html=True
        )
        st.write("### General Wellness Tips")

        st.write(
            """
            - Maintain healthy sleep habits
            - Stay hydrated and eat balanced meals
            - Talk with supportive people
            - Take regular mental breaks
            - Practice mindfulness and relaxation
            - Spend time doing activities you enjoy
            """
        )

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <center>
    <p style='color:gray;'>
    Developed using NLP, TensorFlow, Keras, and Streamlit
    </p>
    </center>
    """,
    unsafe_allow_html=True
)


