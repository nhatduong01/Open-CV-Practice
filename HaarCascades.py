'''Face detection is different from Face regconition.
Haar Cascede is used to detect faces in an image. 
I used the pre-trained model from githubs of Open CV.
The link is : https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml'''
import cv2 as cv
img = cv.imread("Images/cow.jpg")
cv.imshow("Group", img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Group", gray)


haar_cascades = cv.CascadeClassifier('Haar_face_detection.xml')
faces_rectangle = haar_cascades.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=7)
print(f'Number of faces found = {len(faces_rectangle)}')
# We draw a rectangle on a detected face.
for (x, y, w, h) in faces_rectangle:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
cv.imshow("Detected Faces", img)
cv.waitKey(0)
