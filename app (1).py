import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('/content/drive/MyDrive/my_model.hdf5')



# Define a function to preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (28,28))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255
    return img

# Define the Streamlit app
def app():
    # Set the page configuration
    st.set_page_config(page_title="Tuber-detect", page_icon="https://cdn.icon-icons.com/icons2/2823/PNG/512/chest_xray_icon_179657.png", layout="wide")
    st.title("TB Chest Radiography Diagnosis")
    st.write("Upload a chest X-ray image and get a diagnosis of whether the person has Tuberculosis or not.")

    # Create a file uploader for the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image from the uploaded file
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Preprocess the image
        processed_image = preprocess_image(image)


        # Make a prediction with the model
        prediction = model.predict(np.array([processed_image]))

        # Get the predicted class
        predicted_class = np.argmax(prediction)

        st.image(image, width=240)

        # Define the class labels and colors
        class_labels = ["does not have Tuberculosis", "has Tuberculosis"]
        class_colors = ["green", "red"]
        

        st.markdown(f'<p style="color:{class_colors[predicted_class]};">The person  {class_labels[predicted_class]}.</p>', 
                    unsafe_allow_html=True)
# Run the app
if __name__ == '__main__':
    app()
