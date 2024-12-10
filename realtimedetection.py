import cv2
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
import numpy as np
# from keras_preprocessing.image import load_img
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
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

# Load weights
model.load_weights("facialemotionmodel.h5")


haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam=cv2.VideoCapture(0)
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}


while True:
    ret, im = webcam.read()  # Read frame from webcam
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (p, q, r, s) in faces:
        image = gray[q:q + s, p:p + r]  # Crop the face from the image
        cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)  # Draw rectangle around the face

        # Resize and preprocess the cropped face
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        
        # Predict emotion
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        
        # Display prediction on the image
        cv2.putText(im, '%s' % (prediction_label), (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the output in a window
    cv2.imshow("Emotion Detection", im)

    # Check for 'q' key to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
