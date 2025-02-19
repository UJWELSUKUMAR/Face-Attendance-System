import os
import cv2
import base64
import pickle
import pandas as pd
import datetime
from zipfile import BadZipFile
from .models import Attendance
from keras_facenet import FaceNet
from yoloface import face_analysis
from django.contrib import messages
from scipy.spatial.distance import cosine
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from django.core.files.base import ContentFile

# Initialize FaceNet and YOLO face detector
embedder = FaceNet()
face = face_analysis()

# Paths for storing data
pickle_path = "enrollment"
log_file = "attendance.xlsx"

# Keep track of logged names
logged_names = set()

# Ensure the log file exists and is a valid Excel file
def ensure_valid_log_file():
    if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
        print("Creating a new attendance.xlsx file...")
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_excel(log_file, index=False, engine='openpyxl')
    else:
        try:
            pd.read_excel(log_file, engine='openpyxl')  # Try reading the file
        except (BadZipFile, ValueError):
            print("Invalid or corrupt attendance.xlsx file. Recreating...")
            os.remove(log_file)
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
            df.to_excel(log_file, index=False, engine='openpyxl')

# Run validation before using the log file
ensure_valid_log_file()

# -------------------------------- Recognize Face --------------------------------

def recognize_face(face_embedding, dir_path):
    min_distance = float('inf')
    recognized_person = "UNKNOWN"
    recognition_accuracy = 0

    for file in os.listdir(dir_path):
        if file.endswith('.pkl'):
            stored_embeddings = pickle.load(open(os.path.join(dir_path, file), 'rb'))
            for stored_embedding in stored_embeddings:
                distance = cosine(face_embedding, stored_embedding)
                accuracy = (1 - distance) * 100  

                if distance < min_distance:
                    min_distance = distance
                    recognized_person = file.split('.')[0]  
                    recognition_accuracy = accuracy

    if min_distance < 0.6:
        return recognized_person, recognition_accuracy
    else:
        return "UNKNOWN", 0

# -------------------------------- Log Attendance --------------------------------

def log_attendance(name):
    ensure_valid_log_file()  # Ensure the file is valid before writing

    current_time = datetime.datetime.now()
    current_date = datetime.date.today()
    current_time_of_day = current_time.strftime('%H:%M:%S')

    # Prevent duplicate logging in one session
    if name in logged_names:
        print(f"Attendance for {name.upper()} is already logged in this session.")
        return

    try:
        df = pd.read_excel(log_file, engine='openpyxl')
    except (BadZipFile, ValueError):
        print("Error reading attendance.xlsx. Recreating file...")
        ensure_valid_log_file()
        df = pd.read_excel(log_file, engine='openpyxl')

    # Prevent duplicate name entries on the same date
    if df.loc[(df["Name"] == name) & (df["Date"] == str(current_date))].empty:
        new_entry = pd.DataFrame([[name.upper(), current_date, current_time_of_day]],
                                 columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)

        df.to_excel(log_file, index=False, engine='openpyxl')  # Ensure openpyxl is used
        logged_names.add(name)  # Add to session log
        print(f"Attendance logged for {name.upper()} on {current_date}.")

# -------------------------------- Django Views --------------------------------

def index(request):
    return render(request, 'index.html')

def second(request):
    return render(request, 'second.html')

# -------------------------------- Video Stream --------------------------------

def gen(request):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, boxes, _ = face.face_detection(frame_arr=frame, frame_status=True, model='tiny')

        if len(boxes) > 0:
            for box in boxes:
                x, y, w, h = box
                face_crop = frame[y:y + w, x:x + h]

                if face_crop is not None and face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (160, 160))

                    face_embedding = embedder.embeddings([face_resized])[0]
                    name, accuracy = recognize_face(face_embedding, pickle_path)

                    if name != "UNKNOWN" and accuracy > 68:
                        cv2.putText(frame, f"{name} ({accuracy:.2f}%)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)
                        log_attendance(name)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

def video_stream(request):
    return StreamingHttpResponse(gen(request), content_type='multipart/x-mixed-replace; boundary=frame')

# -------------------------------- Add Person --------------------------------

def add_person(request):
    if request.method == "POST":
        name = request.POST.get("name").strip().title()
        image = request.POST.get("image")  

        if Attendance.objects.filter(name=name).exists():
            messages.error(request, 'This name is already registered.')
            return redirect('second')

        if image:
            format, imgstr = image.split(';base64,')
            ext = format.split('/')[-1]
            image_name = f"{name}.jpg"
            image = ContentFile(base64.b64decode(imgstr), name=image_name)

            attendance = Attendance.objects.create(name=name, image=image)
            attendance.save()

            messages.success(request, 'Person added successfully')
            return redirect('second')
        else:
            messages.error(request, 'No image captured. Please try again.')
            return redirect('second')

    return render(request, "second.html")
