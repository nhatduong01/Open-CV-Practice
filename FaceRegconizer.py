import os
import cv2 as cv
import numpy
people = ["Thao Nguyen", "Emma Watson",
          "Ariana Grande", "Camila Cabello"]
DIR = R'C:\Users\ADMIN\Desktop\Basic of Machine Learning\Open CV\FaceRegconition'
features = []
labels = []


def train():
    for person in people:
        path = os.path.join(DIR, person)
