import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO

def load_image_from_url(url):
    response = requests.get(url)
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def detect_faces(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(1, 1))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return image

def main():
    st.title("Face Detection App")
    st.write("Enter the URL of an image to detect faces:")
    image_url = st.text_input("Image URL:")
    
    if image_url:
        try:
            image = load_image_from_url(image_url)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button('Detect Faces'):
                with st.spinner('Detecting...'):
                    result_image = detect_faces(image)
                    st.image(result_image, caption='Detected Faces', use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

if __name__ == '__main__':
    main()
