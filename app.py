# mongodb credentials
# username = pratik
# password = LSMt1jebHWuPEB0m
import base64
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import cv2
import csv
from datetime import datetime
import pyttsx3
from takeImage import *
from recognizer import face_utils
from trainImage import *
from flask_cors import CORS  # Add this import
from automaticAttedance import *
from render import *
from show_attendance import *

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


@app.route('/register', methods=['GET'])
def register():
    """Render the student registration form"""
    return render_template('register_student.html')

@app.route('/automatic_attendance')
def automatic_attendance():
    # Implement functionality to start automatic attendance
    return render_template('automatic_attendance.html')


@app.route('/api_register_student', methods=['POST'])
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

@app.route('/capture', methods=['POST'])
def capture():
    try:
        data = request.get_json()
        subject = data.get('subject')
        
        # Here you would implement your face recognition logic
        # For demonstration, we'll use mock data
        recognized_student = {
            "student_id": "S12345",  # Replace with actual recognized student ID
            "name": "John Doe"       # Replace with actual recognized student name
        }
        

        print(f"Attendance captured for student ID: {recognized_student['student_id']} in subject: {subject}")

        return jsonify({
            "success": True,
            "student_id": recognized_student["student_id"],
            "name": recognized_student["name"],
            "timestamp": datetime.now().isoformat(),
            "subject": subject
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/save_attendance', methods=['POST'])
def save_attendance():
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        name = data.get('name')
        subject = data.get('subject')
        timestamp = datetime.now()

        if not all([student_id, name, subject]):
            return jsonify({"error": "Missing fields"}), 400

        date_str = timestamp.strftime("%Y-%m-%d")

        # Check if already recorded
        existing = attendance_collection.find_one({
            "date": date_str,
            "subject": subject,
            "student_id": student_id
        })

        if existing:
            return jsonify({"error": "Attendance already recorded"}), 400

        # Save attendance
        attendance_collection.insert_one({
            "student_id": student_id,
            "name": name,
            "subject": subject,
            "status": "Present",
            "timestamp": timestamp,
            "date": date_str
        })

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/automatic_attendance_page')
def automatic_attendance_page():
    return Response(gen_frames(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/view_attendance')
def view_attendance():
    return render_template("attendance_table.html")

@app.route('/get_classes', methods=['POST'])
def get_classes():
    selected_date = request.json.get("date")
    classes = attendance_collection.find({"date": selected_date}).distinct("class")
    return jsonify(classes)

@app.route('/get_attendance', methods=['POST'])
def get_attendance():
    selected_date = request.json.get("date")
    selected_class = request.json.get("class")
    record = attendance_collection.find_one({"date": selected_date, "class": selected_class})
    return jsonify(record.get("students", []))

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)