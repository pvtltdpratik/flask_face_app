# mongodb credentials
# username = pratik
# password = LSMt1jebHWuPEB0m
import base64
import pandas as pd
from flask import Flask, render_template, request, Response,redirect, url_for, flash, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import cv2
import csv
from datetime import datetime
from PIL import Image
import io
import pyttsx3
from takeImage import *
from recognizer import face_utils
from trainImage import *
from flask_cors import CORS  # Add this import
from automaticAttedance import *
from render import *
from show_attendance import *
from deepface import DeepFace
import os
app  = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for flash messages
CORS(app)
face_recognizer = face_utils.FaceRecognizer()

# Path to save images and student details
haarcasecade_path = "./haarcascade_frontalface_default.xml"
trainimage_path = "/training_images/"
studentdetail_path = "./StudentDetails/studentdetails.csv"
trainimageLabel_path = "./TrainingImageLabel/Trainner.yml"

attendance_data = {}


# MongoDB connection
# MongoDB Setup
client = MongoClient(os.getenv('MONGO_URI'))
db = client["student_database"]
students_collection = db["students"]
db = client["attendance_db"]
attendance_collection = db["attendance_records"]
# Configuration
UPLOAD_FOLDER = 'static/student_photos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function for text-to-speech
import pyttsx3

def text_to_speech(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")


@app.route('/')
def index():
    return render_template('index.html')  # Render your index page


@app.route(os.getenv('API_URI') + '/register', methods=['GET'])
def register():
    """Render the student registration form"""
    return render_template('register_student.html')

@app.route(os.getenv('API_URI') + '/automatic_attendance')
def automatic_attendance():
    # Implement functionality to start automatic attendance
    return render_template('automatic_attendance.html')


@app.route(os.getenv('API_URI') + '/api_register_student', methods=['POST'])
def api_register_student():
    try:
        # Get form data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        enrollment_no = data.get('enrollment_no')
        name = data.get('name')
        class_name = data.get('class')
        date_of_birth = data.get('date_of_birth')
        image_data = data.get('image_data')

        # Validate required fields
        if not all([enrollment_no, name, class_name, date_of_birth, image_data]):
            return jsonify({'error': 'All fields are required'}), 400

        # Process image data
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)

        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{enrollment_no}_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, 'wb') as f:
            f.write(binary_data)

        # Generate face encoding (for future recognition)
        image = face_recognition.load_image_file(filepath)
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            os.remove(filepath)  # Remove the image if no face detected
            return jsonify({'error': 'No face detected in the image. Please try again.'}), 400

        # In a real application, you would save the encoding to a database
        face_encoding = face_encodings[0].tolist()  # Convert numpy array to list

        # Here you would typically save to a database
        student_data = {
            'enrollment_no': enrollment_no,
            'name': name,
            'class': class_name,
            'date_of_birth': date_of_birth,
            'image_path': filepath,
            'face_encoding': face_encoding
        }
        result = students_collection.insert_one(student_data)
        # For this example, we'll just return success

        return jsonify({
            'success': True,
            'message': 'Student registered successfully',
            'enrollment_no': enrollment_no,
            'filename': filename
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Decode base64 image and convert to numpy array
def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)

@app.route(os.getenv('API_URI') + '/capture', methods=['POST'])
def capture():
    # Step 1: Capture frame from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Failed to capture image"}), 500

    # Step 2: Save captured image temporarily
    os.makedirs("temp", exist_ok=True)
    temp_path = "temp/captured.jpg"
    cv2.imwrite(temp_path, frame)

    try:
        # Step 3: Search in the known student database
        result = DeepFace.find(
            img_path=temp_path,
            db_path="static/student_photos",
            model_name="VGG-Face",  # Or try "Facenet", "ArcFace", "SFace"
            enforce_detection=False
        )

        if result[0].empty:
            return jsonify({"message": "No recognized students in frame"}), 404

        # Step 4: Get matched file name
        matched_img_path = result[0].iloc[0]['identity']
        filename = os.path.basename(matched_img_path)  # e.g., 938_20250506_114831.jpg
        enrollment_no = filename.split("_")[0]          # extract "938"

        # Step 5: Fetch student from MongoDB
        student = students_collection.find_one({"enrollment_no": enrollment_no})
        if not student:
            return jsonify({"message": "Student not found in database"}), 404

        # Step 6: Return matched student info
        return dumps(student), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route(os.getenv('API_URI') + '/automatic_attendance_page')
def automatic_attendance_page():
    return Response(gen_frames(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route(os.getenv('API_URI') + '/view_attendance')
def view_attendance():
    return render_template("attendance_table.html")

@app.route(os.getenv('API_URI') + '/get_classes', methods=['POST'])
def get_classes():
    selected_date = request.json.get("date")
    classes = attendance_collection.find({"date": selected_date}).distinct("class")
    return jsonify(classes)

@app.route(os.getenv('API_URI') + '/get_attendance', methods=['POST'])
def get_attendance():
    selected_date = request.json.get("date")
    selected_class = request.json.get("class")
    record = attendance_collection.find_one({"date": selected_date, "class": selected_class})
    return jsonify(record.get("students", []))

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)