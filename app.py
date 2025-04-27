import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

app = Flask(__name__)
model = YOLO("best_model.pt")  # Load YOLOv8 model

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        if filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}:
            return detect_image(filepath)
        elif filename.rsplit('.', 1)[1].lower() in {'mp4', 'mov'}:
            return detect_video(filepath)
    # Add a return statement for the case where the file is not allowed or other conditions are not met
    return 'Invalid file type or upload error'


def detect_image(filepath):
    img = Image.open(filepath)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert image to np.array and then to BGR format
    results = model(img)  # Perform inference on the image

    # Access the first result from the list and save it
    result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')

    # Use the `save()` method properly by specifying the filename and saving to the upload folder
    results[0].save(filename=result_filepath)  # Save the result without 'save_dir'

    return send_from_directory(app.config['UPLOAD_FOLDER'], 'result.jpg')


def detect_video(filepath):
    cap = cv2.VideoCapture(filepath)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 output
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_video.mp4')  # Output as .mp4
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    if not cap.isOpened():
        return 'Error: Video file could not be opened.'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        results = model(frame)
        annotated_frame = results[0].plot()

        # Write the frame with detection back to a new video file
        out.write(annotated_frame)

        # Display the frame with detections
        cv2.imshow('frame', annotated_frame)

        # If 'q' is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return send_from_directory(app.config['UPLOAD_FOLDER'], 'output_video.mp4')


if __name__ == '__main__':
    app.run(debug=True)
    