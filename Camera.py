import cv2
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from typing import Optional


class MoodMusicRecommender:
    def __init__(self, model_path: str, music_data_path: str):
        """Initialize the recommender with the model and music data."""
        self.model = load_model(model_path)
        self.mood_music = pd.read_csv(music_data_path)

        # Mapping of emotions to moods
        self.emotion_mapping = {
            0: 'Calm',  # angry
            1: 'Calm',  # disgust
            2: 'Calm',  # fear
            3: 'Happy',  # happy
            4: 'Happy',  # neutral
            5: 'Sad',    # sad
            6: 'Energetic'  # surprise
        }

    def get_recommendations(self, emotion_idx: int, num_songs: int = 5) -> Optional[pd.DataFrame]:
        """Get music recommendations based on the predicted emotion."""
        mood = self.emotion_mapping[emotion_idx]

        # Get recommendations
        mood_songs = self.mood_music[self.mood_music['mood'] == mood]
        if mood_songs.empty:
            raise ValueError(f"No songs available for the mood '{mood}'.")

        return mood_songs.sample(n=min(num_songs, len(mood_songs)))[['name', 'artist', 'mood']]


# Initialize the facial emotion model
model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(7, activation='softmax'),
])


model.load_weights("facialemotionmodel.h5")

recommender = MoodMusicRecommender("facialemotionmodel.h5", "music_data.csv")

# Haarcascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Helper function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start the webcam for real-time emotion detection
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

capturing_image = False  
captured_face = None     

while True:
    ret, im = webcam.read()  
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  

    if capturing_image and len(faces) > 0:
        
        for (p, q, r, s) in faces:
            captured_face = gray[q:q + s, p:p + r]
            cv2.imwrite('captured_image.jpg', captured_face)
            print("Image captured and saved as 'captured_image.jpg'")

            
            captured_face_resized = cv2.resize(captured_face, (48, 48))
            img = extract_features(captured_face_resized)

            
            pred = model.predict(img)
            emotion_idx = pred.argmax()
            prediction_label = labels[emotion_idx]

            
            recommendations = recommender.get_recommendations(emotion_idx)
            print("\nMusic Recommendations based on emotion '%s':" % prediction_label)
            print(recommendations)

    
            cv2.putText(im, '%s' % (prediction_label), (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)  # Draw rectangle around the face

        # Reset capturing_image flag
        capturing_image = False  

    # Show the current webcam frame
    cv2.imshow("Emotion Detection", im)

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('c') and not capturing_image:
        
        capturing_image = True
        print("Press 'c' again to capture image...")

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
