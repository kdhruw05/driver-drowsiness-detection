import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance as dist
from pygame import mixer
import tkinter as tk
from tkinter import simpledialog

mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
flag = 0
frame_check = 20
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

running = False

def start_capture():
    global running
    running = True
    while running:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            x1, y1, x2, y2 = (subject.left(), subject.top(), subject.right(), subject.bottom())
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEar = eye_aspect_ratio(leftEye)
            rightEar = eye_aspect_ratio(rightEye)
            ear = (leftEar + rightEar) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "************ALERT***********", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def stop_capture():
    global running
    running = False

def set_threshold():
    global thresh
    new_thresh = simpledialog.askfloat("Input", "Set new threshold (0.1 to 0.5):", minvalue=0.1, maxvalue=0.5)
    if new_thresh is not None:
        thresh = new_thresh

root = tk.Tk()
root.title("Drowsiness Detector")

start_button = tk.Button(root, text="Start", command=start_capture)
start_button.pack()

stop_button = tk.Button(root, text="Stop", command=stop_capture)
stop_button.pack()

threshold_button = tk.Button(root, text="Set Threshold", command=set_threshold)
threshold_button.pack()

root.mainloop()

cap.release()
cv2.destroyAllWindows()