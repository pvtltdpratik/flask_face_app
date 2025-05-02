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

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for flash messages

# Path to save images and student details
haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimage_path = "./TrainingImage"
studentdetail_path = "./StudentDetails/studentdetails.csv"

attendance_data = {}


# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["student_database"]
students_collection = db["students"]

# Function for text-to-speech
def text_to_speech(user_text):
    engine = pyttsx3.init()
    engine.say(user_text)
    engine.runAndWait()


@app.route('/')
def index():
    return render_template('index.html')  # Render your index page

@app.route('/register_student', methods=['POST'])
def register_student():
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['full_name', 'class', 'date_of_birth']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Prepare student document
        student = {
            "full_name": data["full_name"],
            "class": data["class"],
            "date_registered": datetime.utcnow(),  # Auto-set current datetime
            "date_of_birth": datetime.strptime(data["date_of_birth"], "%Y-%m-%d"),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Insert into MongoDB
        result = students_collection.insert_one(student)
        
        # Return success response
        return jsonify({
            "message": "Student registered successfully",
            "student_id": str(result.inserted_id)
        }), 201
        
    except ValueError as e:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/automatic_attendance')
def automatic_attendance():
    # Implement functionality to start automatic attendance
    return render_template('automatic_attendance.html')


@app.route('/view_attendance')
def view_attendance():
    # Implement functionality to view the attendance
    return "View Attendance Page"


@app.route('/fill_attendance', methods=['POST'])
def fill_attendance():
    subject = request.form.get('subject')
    if not subject:
        return render_template('error.html', message="Please enter a subject name.")

    return render_template('attendance_form.html', subject=subject)  # Form for entering student data


@app.route('/submit_attendance', methods=['POST'])
def submit_attendance():
    subject = request.form.get('subject')
    enrollment = request.form.get('enrollment')
    student_name = request.form.get('student_name')

    if not enrollment or not student_name:
        return render_template('error.html', message="Please enter both enrollment number and student name.")

    # Store the attendance data in a dictionary
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    if subject not in attendance_data:
        attendance_data[subject] = []

    attendance_data[subject].append({
        "Enrollment": enrollment,
        "Name": student_name,
        "Date": date,
    })

    return render_template('attendance_form.html', subject=subject, message="Data entered successfully!")


@app.route('/create_csv', methods=['POST'])
def create_csv():
    subject = request.form.get('subject')
    if subject not in attendance_data:
        return render_template('error.html', message="No data available for this subject.")

    date = datetime.datetime.now().strftime("%Y_%m_%d")
    csv_filename = f"Attendance/{subject}_{date}.csv"

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(attendance_data[subject])
    df.to_csv(csv_filename, index=False)

    return render_template('success.html', message="CSV created successfully!")


@app.route('/check_sheets', methods=['GET'])
def check_sheets():
    # List all saved CSV files
    files = os.listdir('Attendance/')
    return render_template('files_list.html', files=files)


if __name__ == '__main__':
    app.run(debug=True)