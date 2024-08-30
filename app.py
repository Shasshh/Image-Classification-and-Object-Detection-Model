from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS

from main import preprocessed_frame

app = Flask(__name__)
CORS(app)

# Load your pre-trained model
model = load_model('C:\\Users\\Asus\\PycharmProjects\\Live\\results')

input_size = (224, 224)
# Function to preprocess the image
def preprocess_image(frame):
    # Your preprocessing logic here
    resized_frame = cv2.resize(frame, input_size)
    normalized_frame = resized_frame / 255.0
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
    return preprocessed_frame

# Function to perform predictions on the preprocessed image
def predict(frame):
    preprocessed_frame = preprocess_image(frame)
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Function to capture frames from the camera
def camera():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        # Perform prediction on the frame
        predicted_class = predict(frame)
        # Overlay the prediction on the frame
        cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
    app.run(debug=True)
