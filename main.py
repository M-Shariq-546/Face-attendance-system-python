import face_recognition as fc
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#Here we just gove the pictures to detect the students

muji_image = fc.load_image_file("photos/muji.jpeg")
muji_encoding = fc.face_encodings(muji_image)[0]

hamza_image = fc.load_image_file("photos/hamza.jpeg")
hamza_encoding = fc.face_encodings(hamza_image)[0]

moeez_image = fc.load_image_file("photos/moeez.jpeg")
moeez_encoding = fc.face_encodings(moeez_image)[0]

shariq_image = fc.load_image_file("photos/shariq.jpeg")
shariq_encoding = fc.face_encodings(shariq_image)[0]

#known faces images

known_faces_images = [
muji_encoding,
hamza_encoding,
moeez_encoding,
shariq_encoding
]

#names of students whose pictures are provided

known_faces_names = [
"Mujtaba Tahir" , 
"Moeez Ahmad",
"Hamza Ayub",
"Muhammad Shariq Shafiq"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names =[]
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


f = open(current_date+'.csv' , 'w+' , newline= '')
lnwriter = csv.writer(f)

#All the given code is to locate the image and detect it

while True:
    _,frame= video_capture.read()
    small_frame = cv2.resize(frame,(0,0), fx=0.25 , fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_location = fc.face_locations(rgb_small_frame)
        face_encodings = fc.face_encodings(rgb_small_frame , face_location)
        face_names = []
        for face_encoding in face_encodings:
            matches = fc.compare_faces(known_faces_images , face_encoding)
            name = ""
            face_distance = fc.face_distance(known_faces_images , face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
            
            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name , current_time])
            
    cv2.imshow(" Facial Attendance System ", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
f.close()

#close all windows and files and let's go to execution