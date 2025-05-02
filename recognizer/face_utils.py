import os
import cv2
import numpy as np
from PIL import Image
import csv

class FaceRecognizer:
    def __init__(self):
        self.haarcascade_path = "../haarcascade_frontalface_default.xml"
        self.train_image_path = "../training_images/"
        self.train_image_label_path = "../TrainingImageLabel/Trainner.yml"
        self.student_details_path = "../StudentDetails/studentdetails.csv"
        
        # Create directories if they don't exist
        os.makedirs(self.train_image_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.train_image_label_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.student_details_path), exist_ok=True)

    def take_images(self, enrollment, name, stdclass, dob):
        """Capture face images for registration"""
        if not enrollment or not name:
            return "Please enter both Enrollment Number and Name"
        
        try:
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier(self.haarcascade_path)
            sample_num = 0
            directory = f"{enrollment}_{name}"
            path = os.path.join(self.train_image_path, directory)
            
            if os.path.exists(path):
                return "Student data already exists"
                
            os.mkdir(path)
            
            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sample_num += 1
                    cv2.imwrite(
                        os.path.join(path, f"{name}_{enrollment}_{sample_num}.jpg"),
                        gray[y:y+h, x:x+w]
                    )
                    cv2.imshow("Registering Face", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q') or sample_num > 50:
                    break
                    
            cam.release()
            cv2.destroyAllWindows()
            
            # Save student details
            with open(self.student_details_path, 'a+', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([enrollment, name])
            
            return f"Images saved for ER No: {enrollment} Name: {name}"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def train_model(self):
        """Train the face recognition model"""
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier(self.haarcascade_path)
            faces, ids = self._get_images_and_labels()
            recognizer.train(faces, np.array(ids))
            recognizer.save(self.train_image_label_path)
            return "Model trained successfully"
        except Exception as e:
            return f"Training error: {str(e)}"

    def _get_images_and_labels(self):
        """Helper function to get training images and labels"""
        image_paths = []
        for d in os.listdir(self.train_image_path):
            dir_path = os.path.join(self.train_image_path, d)
            if os.path.isdir(dir_path):
                for f in os.listdir(dir_path):
                    if f.endswith('.jpg'):
                        image_paths.append(os.path.join(dir_path, f))
        
        faces = []
        ids = []
        
        for image_path in image_paths:
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            student_id = int(os.path.split(image_path)[-1].split('_')[1])
            faces.append(image_np)
            ids.append(student_id)
            
        return faces, ids