import streamlit as  st
import numpy as np
import pandas as pd
import joblib
import pickle
import tensorflow as tf
import altair as alt
from datetime import datetime
import plotly.express as px 
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table
import sqlite3


with sqlite3.connect('data/data.db') as conn:
    c = conn.cursor()
    create_emotionclf_table(c)
    create_page_visited_table(c)

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


            with sqlite3.connect('data/data.db') as conn:
                c = conn.cursor()
                add_prediction_details(raw_text,prediction,np.max(probability),datetime.now(), c, conn)

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
        with sqlite3.connect('data/data.db') as conn:
            add_page_visited_details("Monitor",datetime.now(), conn.cursor(), conn)
            st.subheader("Monitor App")

            with st.expander("Page Metrics"):
                
                page_visited_details = pd.DataFrame(view_all_page_visited_details(conn.cursor()),columns=['Pagename','Time_of_Visit'])
                st.dataframe(page_visited_details)	

                pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
                c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
                st.altair_chart(c,use_container_width=True)	

                p = px.pie(pg_count,values='Counts',names='Pagename')
                st.plotly_chart(p,use_container_width=True)

            with st.expander('Emotion Classifier Metrics'):
                df_emotions = pd.DataFrame(view_all_prediction_details(conn.cursor()),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
                st.dataframe(df_emotions)

                prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
                pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
                st.altair_chart(pc,use_container_width=True)	

    else:
        st.subheader("About")
        add_page_visited_details("About",datetime.now())


if __name__ == "__main__":
    main()