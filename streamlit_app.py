import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect_faces(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(1, 1))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return image

def main():
    st.title("Face Detection App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Detect Faces'):
            with st.spinner('Detecting...'):
                result_image = detect_faces(image)
                st.image(result_image, caption='Detected Faces', use_column_width=True)

if __name__ == '__main__':
    main()
