from flask import Flask, render_template, Response
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load your pre-trained model
model = load_model('C:\\Users\\Asus\\PycharmProjects\\Live\\results')

# Define the input size expected by your model
input_size = (224, 224)  # Adjust this based on your model's input size

# Define the class labels and names corresponding to the output classes of your model
class_names = [
    (0, "Bijapur Jowar"),
    (1, "Masoor Dal"),
    (2, "Moong Dal"),
    (3, "Moth Bean"),
    (4, "Peanut"),
    (5, "Putani"),
    (6, "Rice"),
    (7, "Toor Dal"),
    (8, "Urad Dal"),
    (9, "Wheat"),
    (10, "Whole Moong")
    # Add more tuples for additional classes
]

# Function to preprocess the image
def preprocess_image(frame):
    resized_frame = cv2.resize(frame, input_size)
    normalized_frame = resized_frame / 255.0
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
    return preprocessed_frame

# Function to perform predictions on the preprocessed image
def predict(frame):
    preprocessed_frame = preprocess_image(frame)
    predictions = model.predict(preprocessed_frame)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label, predicted_class_name = class_names[predicted_class_index]
    return predicted_class_label, predicted_class_name

# Function to capture frames from the camera
def camera():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        # Perform prediction on the frame
        predicted_class_label, predicted_class_name = predict(frame)
        # Overlay the prediction on the frame
        text = f'Predicted Class: {predicted_class_label} - {predicted_class_name}'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to the live camera feed
@app.route('/video_feed')
def video_feed():
    return Response(camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main entry point
if __name__ == '__main__':
    app.run(port=5000, debug=True)




#http://127.0.0.1:5000/video_feed
#IPv6 Address : 2405:201:d00e:505e:2c7c:9b95:7327:f17f