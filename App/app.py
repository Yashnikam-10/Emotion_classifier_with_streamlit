import streamlit as  st
import numpy as np
import pandas as pd
import joblib
import pickle
import tensorflow as tf
import altair as alt

# Load class names
classNames = np.load("data/class_names.npy")

# Load tokenizer pickle file
with open('data/tokenizer.pickle', 'rb') as handle:
        Tokenizer = pickle.load(handle)

# Load model
model = tf.keras.models.load_model("data/final_model")

def predict_emotions(sentence):
    sentence_processed = Tokenizer.texts_to_sequences([sentence])
    sentence_processed = np.array(sentence_processed)
    sentence_padded = tf.keras.preprocessing.sequence.pad_sequences(sentence_processed, padding='post', maxlen=100)

    result = model.predict(sentence_padded)
    return classNames[np.argmax(result)], result

def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home-Emotion In Text")

        with st.form(key = 'emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")

        if submit_text:
            col1, col2 = st.columns(2)

            prediction, probability = predict_emotions(raw_text)
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                st.write(prediction)

            with col2:
                st.success("Prediction Probability")
                # st.write(probability)

                output_df = pd.DataFrame(probability, columns=[name for name in classNames])
                # st.write(output_df.T)
                output_df_clean = output_df.T.reset_index()
                output_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(output_df_clean).mark_bar().encode(x="emotions", y="probability", color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        st.subheader("Monitor App")

    else:
        st.subheader("About")

if __name__ == "__main__":
    main()