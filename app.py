# mongodb credentials
# username = pratik
# password = LSMt1jebHWuPEB0m
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import cv2
import csv
from datetime import datetime
import pyttsx3
from takeImage import TakeImage
from recognizer import face_utils
from  takeImage import TakeImage
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


@app.route('/api/register_student', methods=['POST'])
def register_student():
    try:
        # Handle both JSON and form data
        student_name = request.form.get('name')
        enrollment = request.form.get('enrollment_no')
        student_class = request.form.get('class')
        date_of_birth = request.form.get('date_of_birth')

        data = {
            'full_name': student_name,
            'enrollment': enrollment,
            'class': student_class,
            'date_of_birth': date_of_birth,
        }

        TakeImage('19', 'Pratik', haarcasecade_path, trainimage_path, "All are required", "",text_to_speech("error occured for taking images"))
        result = students_collection.insert_one(data)
        return jsonify({
            "message": "Student registered successfully",
            "student_id": str(result.inserted_id)
        }), 201
    except ValueError as e:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/takeImageFunction')
def takeImageFunction():
    result = TakeImage('19', 'Pratik', haarcasecade_path, trainimage_path, "All are required", "",text_to_speech("error occured for taking images"))
    flash(result)
    text_to_speech(result)


@app.route('/trainImageFunction')
def trainImageFunction():
    result = TrainImage(haarcasecade_path, trainimage_path, trainimageLabel_path, "All are required",text_to_speech("error occured for taining images"))
    flash(result)
    text_to_speech(result)


@app.route('/automatic_attendance')
def automatic_attendance():
    # Implement functionality to start automatic attendance
    return render_template('automatic_attendance.html')


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

from flask import Flask, render_template, request, redirect, url_for, flash

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)