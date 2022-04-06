


# imports
# 
import mediapipe as mp
import cv2
import pandas as pd
import numpy as np


# create holistics instances
# 
mp_drawing_utils = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles
landmarks = ['class']


# create a landmarks csv
# 
def landmark_csv():
    global landmarks
    for i in range(1,469):
        landmarks += ['x{}'.format(i),'y{}'.format(i),'z{}'.format(i),'v{}'.format(i)]
    # print(len(landmarks))
    return pd.DataFrame(columns=landmarks), landmarks


# capture frames
# 
def capture_frames( target, n_frames = 20):
    
    track_frames = 0 
    cap=cv2.VideoCapture(0)
    work, landmarks = landmark_csv()
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            _,frame=cap.read()
            
            image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=holistic.process(image)
            # print(result.face_landmarks)
            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            mp_drawing_utils.draw_landmarks(
            image,
            result.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing_utils.draw_landmarks(image,result.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
            mp_drawing_utils.draw_landmarks(image,result.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
            mp_drawing_utils.draw_landmarks(image,result.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
            #image = cv2.resize(image, (1000,7000))
            cv2.imshow('raw',image)
            
            try:
                face=np.array(result.face_landmarks.landmark)
                face_row=list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in face]).flatten())
                face_row.insert(0, target)
                added=pd.DataFrame([face_row],columns=landmarks)
                work=pd.concat([work,added])
                track_frames += 1
            except:
                pass
            
            if (cv2.waitKey(10) & 0xFF == ord('Q')) or track_frames > n_frames :
                break
    # destroy frame window
    cap.release()
    cv2.destroyAllWindows()
    
    return work


