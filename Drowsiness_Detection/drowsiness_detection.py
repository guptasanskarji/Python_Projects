import face_recognition
import cv2
from scipy.spatial import distance as dist
import playsound
# from threading import Thread
from multiprocessing import Process
import numpy as np

MIN_EAR = 0.30
EYE_AR_COSEC_FRAMES = 10

COUNTER = 0
ALARM_ON = False
def sound_alarm(soundfile):
    playsound.playsound(soundfile)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0],eye[3])
    ear = (A+B)/(2*C)
    return ear

def main():
    global COUNTER, ALARM_ON
    video_capture = cv2.VideoCapture(0)
    # video_capture.set(3,320)
    # video_capture.set(4,240)
    while True:
        ret, frame = video_capture.read()
        face_landmarks_list = face_recognition.face_landmarks(frame)
        for face_landmark in face_landmarks_list:
            leftEye = face_landmark['left_eye']
            rightEye = face_landmark['right_eye']

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR)/2

            lpts = np.array(leftEye)
            rpts = np.array(rightEye)

            # cv2.polylines(frame,[lpts],True,(255,255,0),1)
            # cv2.polylines(frame,[rpts],True,(255,255,0),1)

            if ear<MIN_EAR:
                COUNTER+=1
                if COUNTER>=EYE_AR_COSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Process(target=sound_alarm, args=('C:\\Users\\HP\\Desktop\\Python Projects\\Drowsiness_Detection\\alarm.mp3',))
                        t.daemon = True
                        t.start()
                cv2.putText(frame,"Alert!! you are feeling asleep!",(5,10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
            else:
                COUNTER=0
                ALARM_ON=False
        cv2.putText(frame,"EAR {: .2f}".format(ear),(300,10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        cv2.imshow("Sleep Detection", frame)
        if cv2.waitKey(1)==ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# import mediapipe as mp
# import cv2
# import time
# from scipy.spatial import distance as dist
# import playsound
# from threading import Thread
# import numpy as np

# MIN_AER = 0.30
# EYE_AR_COSEC_FRAMES = 10

# COUNTER = 0
# ALARM_ON = False

# def sound_alarm(soundfile):
#     playsound.playsound(soundfile)

# def eye_aspect_ratio(eye):
#     A = dist.euclidean(eye[1],eye[5])
#     B = dist.euclidean(eye[2],eye[4])
#     C = dist.euclidean(eye[0],eye[3])
#     ear = (A+B)/(2*C)
#     return ear

# def main():
#     global COUNTER, ALARM_ON
#     video_capture = cv2.VideoCapture(0)
#     video_capture.set(3,320)
#     video_capture.set(4,240)
#     face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
#     while True:
#         ret, frame = video_capture.read()
#         # face_landmarks_list = face_recognition.face_landmarks(frame) 
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
#         output = face_mesh.process(rgb_frame)
#         face_landmarks_list = output.multi_face_landmarks
#         for face_landmark in face_landmarks_list:
#             leftEye = face_landmark['left_eye']
#             rightEye = face_landmark['right_eye']

#             leftEAR = eye_aspect_ratio(leftEye)
#             rightEAR = eye_aspect_ratio(rightEye)
#             ear = (leftEAR + rightEAR)/2

#             lpts = np.array(leftEye)
#             rpts = np.array(rightEye)

#             cv2.polylines(frame,[lpts],True,(255,255,0),1)
#             cv2.polylines(frame,[rpts],True,(255,255,0),1)

#             if ear<MIN_AER:
#                 COUNTER+=1
#                 if COUNTER>=EYE_AR_COSEC_FRAMES:
#                     if not ALARM_ON:
#                         ALARM_ON = True
#                         t = Thread(target=sound_alarm, args=('alarm.wav'))
#                         t.daemon = True
#                         t.start()
#                 cv2.putText(frame,"Alert!! you are feeling asleep!",(5,10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
#             else:
#                 COUNTER=0
#                 ALARM_ON=False
#         cv2.putText(frame,"EAR {: .2f}".format(ear),(300.10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
#         cv2.imshow("Sleep Detection", frame)
#         if cv2.waitKey(1)==ord('q'):
#             break
#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()