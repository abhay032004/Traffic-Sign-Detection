import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load YOLOv8 model once during startup
model = YOLO("best_model.pt")

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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

        file_ext = filename.rsplit('.', 1)[1].lower()
        if file_ext in {'jpg', 'jpeg', 'png'}:
            return detect_image(filepath)
        elif file_ext in {'mp4', 'mov'}:
            return detect_video(filepath)

    return 'Invalid file type or upload error'


def detect_image(filepath):
    img = Image.open(filepath)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = model(img)

    result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    results[0].save(filename=result_filepath)

    return send_from_directory(app.config['UPLOAD_FOLDER'], 'result.jpg')


def detect_video(filepath):
    cap = cv2.VideoCapture(filepath)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_video.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    if not cap.isOpened():
        return 'Error: Video file could not be opened.'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    return send_from_directory(app.config['UPLOAD_FOLDER'], 'output_video.mp4')


if __name__ == '__main__':
    app.run(debug=True)
