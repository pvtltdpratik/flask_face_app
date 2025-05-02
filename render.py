from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import face_recognition
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Configuration
STUDENT_PHOTOS_DIR = 'static/student_photos'
ATTENDANCE_DIR = 'static/attendance'
os.makedirs(STUDENT_PHOTOS_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Global variables
latest_frame = None
known_face_encodings = []
known_face_names = []


# Load known student faces
def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(STUDENT_PHOTOS_DIR):
        if filename.endswith(('.jpg', '.png')):
            image = face_recognition.load_image_file(f"{STUDENT_PHOTOS_DIR}/{filename}")
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_frame(self):
        global latest_frame
        ret, frame = self.cap.read()
        if ret:
            latest_frame = frame.copy()
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        return None

    def __del__(self):
        self.cap.release()


def gen_frames(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    load_known_faces()  # Refresh known faces
    return render_template('index.html', students=known_face_names)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    global latest_frame
    if latest_frame is None:
        return jsonify({'error': 'No frame available'}), 400

    # Process the frame
    small_frame = cv2.resize(latest_frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    if not face_encodings:
        return jsonify({'error': 'No faces detected'}), 400

    # Compare with known faces
    matches = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        if True in matches:
            matched_index = matches.index(True)
            student_id = known_face_names[matched_index]

            # Save attendance record
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ATTENDANCE_DIR}/{student_id}_{timestamp}.jpg"
            cv2.imwrite(filename, latest_frame)

            return jsonify({
                'success': True,
                'student_id': student_id,
                'timestamp': timestamp
            })

    return jsonify({'error': 'No matching student found'}), 400


if __name__ == '__main__':
    load_known_faces()  # Initial load
    app.run(debug=True)