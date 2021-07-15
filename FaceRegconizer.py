import os
import cv2 as cv
import numpy as np
people = ["Thao Nguyen", "Emma Watson",
          "Ariana Grande", "Camila Cabello"]
DIR = R'C:\Users\ADMIN\Desktop\Basic of Machine Learning\Open CV\FaceRegconition'
features = []
labels = []
haar_cascades = cv.CascadeClassifier('Haar_face_detection.xml')


def train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            img_array = cv.imread(image_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascades.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=7)
            for (x, y, w, h) in faces_rect:
                # Grab the region of interest
                faces_crop = gray[y:y+h, x:x+w]
                features.append(faces_crop)
                labels.append(label)


train()
print("-------Training done ----------")
face_regconizer = cv.face.LBPHFaceRecognizer_create()
features = np.array(features, dtype='object')
labels = np.array(labels)
# Train the regconizer on the features and labels
face_regconizer.train(features, labels)

face_regconizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
