from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
import cv2
import os
import pickle
from mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet 


os.makedirs("enrollment", exist_ok=True)


mtcnn = MTCNN()
embedder = FaceNet()

class Attendance(models.Model):
    name = models.CharField(max_length=50)
    image = models.ImageField(upload_to="attendance_images/") 

# ----------------------------------------------------------

def preprocess_and_extract_embeddings(image_path):
    embeddings = []
    
    if image_path.endswith(('.jpg', '.jpeg', '.png')):
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = mtcnn.detect_faces(rgb_image)

        if detections:
            for detection in detections:
                x, y, w, h = detection['box']
                face = rgb_image[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (160, 160))
                embedding = embedder.embeddings([face_resized])[0]
                embeddings.append(embedding)
    
    return embeddings

def save_embeddings(embeddings, file_name):
    folder = "enrollment"
    os.makedirs(folder, exist_ok=True)  
    file_path = os.path.join(folder, f'{file_name}.pkl')  

    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

@receiver(post_save, sender=Attendance)
def generate_embeddings_for_faculty(sender, instance, created, **kwargs):
    if created: 
        image_path = instance.image.path
        embeddings = preprocess_and_extract_embeddings(image_path)
        file_name = instance.name.replace(" ", "_").lower()
        save_embeddings(embeddings, file_name)
