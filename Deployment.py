import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Preprocessor import Preprocessor
import tempfile
import os
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load Models
emotion_model = load_model('Model/FinalEmotions_Model.h5')
eye_gaze_model = load_model('Model/EyeGaze_Model.h5')

video_int = False
# Set the page title
st.title("Facial Expression and Eye Gaze Detection")

# Video upload functionality
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"], key="expression_video")

if uploaded_video is not None:
    st.video(uploaded_video)
    st.success("Video uploaded successfully!")
    
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_video.read())
        temp_video_path = temp_file.name

        video_int = True


# Create tabs
tab1, tab2, tab3 = st.tabs(["Facial Expression Detection", "Eye Gaze Detection", "Results"])

with tab1:
    if video_int == False:
        st.write("Upload a video to view the emotion distribution.")
    else:
        # Process the uploaded video using Preprocessor
        preprocessor = Preprocessor()
        preprocessed_data = preprocessor.preprocessFER(temp_video_path)

        # Predict emotions for each frame
        processed_frames = np.array(preprocessed_data)
        predictions = emotion_model.predict(processed_frames)
        predicted_emotions = np.argmax(predictions, axis=1)

        # Map predictions to emotion labels
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_counts = pd.Series(predicted_emotions).value_counts().sort_index()
        emotion_counts.index = [emotion_labels[i] for i in emotion_counts.index]

        st.header("Facial Expression Distribution")
        if uploaded_video is not None and len(predicted_emotions) > 0:
            # Create a bar chart based on the emotion counts
            fig, ax = plt.subplots()
            ax.bar(emotion_counts.index, emotion_counts.values, color=["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#FFD700", "#87CEFA", "#90EE90"])
            ax.set_xlabel('Emotions')
            ax.set_ylabel('Frames')
            ax.set_title('Emotion Distribution')
            ax.set_xticks(range(len(emotion_counts.index)))
            ax.set_xticklabels(emotion_counts.index, rotation=45)

            # Display the bar chart
            st.pyplot(fig)
            st.write("### Emotion Distribution")
            
            # Rename the dataframe columns
            emotion_counts = emotion_counts.reset_index()
            emotion_counts.columns = ['Emotions', 'Frames']
            st.table(emotion_counts)
            
            # Display the message for the maximum emotion
            max_emotion = emotion_counts.loc[emotion_counts['Frames'].idxmax()]['Emotions']
            st.write(f"The facial expression of the candidate is {max_emotion} in this video")

with tab2:
    if video_int == False:
        st.write("Upload a video to view the eye gaze distribution.")
    else:
        # Process the uploaded video using Preprocessor
        preprocessor = Preprocessor()
        preprocessed_eye_data = preprocessor.preprocessEyeGaze(temp_video_path)

        # Predict eye gaze for each frame
        processed_eye_frames = np.array(preprocessed_eye_data)
        eye_predictions = eye_gaze_model.predict(processed_eye_frames)
        predicted_eye_gaze = np.argmax(eye_predictions, axis=1)

        # Map predictions to eye gaze labels
        eye_gaze_labels = ['Close Look', 'Forward Look', 'Left Look', 'Right Look']
        eye_gaze_counts = pd.Series(predicted_eye_gaze).value_counts().sort_index()
        eye_gaze_counts.index = [eye_gaze_labels[i] for i in eye_gaze_counts.index]

        st.header("Eye Gaze Distribution")
        if uploaded_video is not None and len(predicted_eye_gaze) > 0:
            # Create a bar chart based on the eye gaze counts
            fig, ax = plt.subplots()
            ax.bar(eye_gaze_counts.index, eye_gaze_counts.values, color=["#FF9999", "#66B2FF", "#99FF99"])
            ax.set_xlabel('Eye Gaze')
            ax.set_ylabel('Frames')
            ax.set_title('Eye Gaze Distribution')
            ax.set_xticks(range(len(eye_gaze_counts.index)))
            ax.set_xticklabels(eye_gaze_counts.index, rotation=45)

            # Display the bar chart
            st.pyplot(fig)
            st.write("### Eye Gaze Distribution")
            
            # Rename the dataframe columns
            eye_gaze_counts = eye_gaze_counts.reset_index()
            eye_gaze_counts.columns = ['Eye Gaze', 'Frames']
            st.table(eye_gaze_counts)
            
            # Display the message for the maximum eye gaze
            max_eye_gaze = eye_gaze_counts.loc[eye_gaze_counts['Frames'].idxmax()]['Eye Gaze']
            st.write(f"The predominant eye gaze of the candidate is {max_eye_gaze} in this video")

with tab3:
    if video_int == False:
        st.write("Upload a video to view the results.")
    else:
        # Define the scoring matrix
        scoring_matrix = {
            'Happy': {'Forward Look': 10, 'Left Look': 7, 'Right Look': 7, 'Close Look': 5},
            'Neutral': {'Forward Look': 9, 'Left Look': 6, 'Right Look': 6, 'Close Look': 4},
            'Surprise': {'Forward Look': 8, 'Left Look': 5, 'Right Look': 5, 'Close Look': 4},
            'Sad': {'Forward Look': 7, 'Left Look': 5, 'Right Look': 5, 'Close Look': 3},
            'Angry': {'Forward Look': 6, 'Left Look': 4, 'Right Look': 4, 'Close Look': 3},
            'Disgust': {'Forward Look': 6, 'Left Look': 4, 'Right Look': 4, 'Close Look': 3},
            'Fear': {'Forward Look': 6, 'Left Look': 4, 'Right Look': 4, 'Close Look': 3}
        }        
        # Display the detected emotion and eye gaze
        st.write(f"Detected Emotion: {max_emotion}")
        st.write(f"Detected Eye Gaze: {max_eye_gaze}")

        # Calculate the score based on the maximum emotion and eye gaze
        if max_emotion in scoring_matrix and max_eye_gaze in scoring_matrix[max_emotion]:
            score = scoring_matrix[max_emotion][max_eye_gaze]
            st.write(f"Candidate score: {score}")
        else:
            st.write("Unable to calculate the score due to missing data.")