import cv2
import numpy as np

# Load HAAR face classifier
face_classifier	=	cv2.CascadeClassifier('C:/Users/charlesyelia/Desktop/Teja/project/ task_py_files/haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
# Function detects faces and returns the cropped face # If no face detected, it returns the input image

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) faces = face_classifier.detectMultiScale(gray, 1.3, 5)

if faces is (): return None

# Crop all faces found for (x,y,w,h) in faces:
cropped_face = img[y:y+h, x:x+w] return cropped_face
# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input while True:

ret, frame = cap.read()
if face_extractor(frame) is not None: count += 1
face = cv2.resize(face_extractor(frame), (200, 200)) face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# Save file in specified directory with unique name
file_name_path = 'C:/Users/charlesyelia/Desktop/team_faces/' + str(count) + '.jpg' cv2.imwrite(file_name_path, face)

# Put count on images and display live count
cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2) cv2.imshow('Face Cropper', face)

else:
print("Face not found") pass

if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key break

cap.release() cv2.destroyAllWindows() print("Collecting Samples Complete")

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = 'C:/Users/charlesyelia/Desktop/team_faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels Training_Data, Labels = [], []

# Open training images in our datapath # Create a numpy array for training data for i, files in enumerate(onlyfiles):
image_path = data_path + onlyfiles[i]
images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) Training_Data.append(np.asarray(images, dtype=np.uint8)) Labels.append(i)

# Create a numpy array for both training data and labels Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer() # pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

vimal_model = cv2.face_LBPHFaceRecognizer.create() # Let's train our model
vimal_model.train(np.asarray(Training_Data), np.asarray(Labels)) print("Model trained sucessefully")
def email(): import smtplib
from email.mime.multipart import MIMEMultipart from email.mime.text import MIMEText

body = "Mail With respect to Face Recognition from Python..!!!" msg = MIMEText(body)

fromaddr = "tejashwinikottha@gmail.com" toaddr = "tejaswirl690@gmail.com" msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "Face Recognized"
#body = "Hurray Your Face Recognition code working !!!!"" # msg.attach(MIMEText(body, 'plain'))
s = smtplib.SMTP('smtp.gmail.com', 587) s.starttls()
passwd="Tejarl@333" s.login(fromaddr, passwd) #text = msg.as_string() s.send_message(msg) print("Mail Sent")
s.quit()


import cv2
import numpy as np import os
face_classifier	=	cv2.CascadeClassifier('C:/Users/charlesyelia/Desktop/Teja/project/ task_py_files/haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
# Convert image to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) faces = face_classifier.detectMultiScale(gray, 1.3, 5) if faces is ():
return img, []
for (x,y,w,h) in faces: cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2) roi = img[y:y+h, x:x+w]
roi = cv2.resize(roi, (200, 200)) return img, roi
# Open Webcam
cap = cv2.VideoCapture(0) a = False
while True:
ret, frame = cap.read()
image, face = face_detector(frame) try:
face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # Pass face to prediction model
# "results" comprises of a tuple containing the label and the confidence value results = vimal_model.predict(face)

if results[1] < 500:
confidence = int( 100 * (1 - (results[1])/400) )
display_string = str(confidence) + '% Confident it is Aishwarya'

cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1,
(255,120,150), 2)

if confidence > 70:
cv2.putText(image, "Hey Teja", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
(0,255,0), 2)
cv2.imshow('Face Recognition', image ) #pyttsx3.speak("Hello Aishwarya") a=True

break else:
cv2.putText(image, "I dont know you, Who are You?", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
cv2.imshow('Face Recognition', image )

except:
cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1,
(0,0,255), 2)
cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
(0,0,255), 2)
cv2.imshow('Face Recognition', image ) pass

if cv2.waitKey(1) == 13:
#13 is the Enter Key break

cap.release() cv2.destroyAllWindows()

if a is True: email();
print("mail sent !!!")



