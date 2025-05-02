import cv2, datetime, time, pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client['attendance_db']
collection = db['records']

def fill_attendance(subject):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    df = pd.read_csv("StudentDetails/studentdetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    attendance = []
    end_time = time.time() + 20

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 70:
                name = df.loc[df["Enrollment"] == id_]["Name"].values[0]
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                attendance.append({'Enrollment': int(id_), 'Name': name, 'Timestamp': timestamp})
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if time.time() > end_time:
            break
        cv2.imshow("Filling Attendance", img)
        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Remove duplicates by Enrollment
    seen = set()
    unique = []
    for a in attendance:
        if a['Enrollment'] not in seen:
            seen.add(a['Enrollment'])
            unique.append(a)

    # Insert into MongoDB
    for record in unique:
        record['Subject'] = subject
        collection.insert_one(record)

    return f"Attendance for {subject} recorded successfully", unique