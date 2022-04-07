


# imports
# 
import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import time

# create holistics instances
# 
mp_drawing_utils = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles


# create a landmarks csv
# 
def landmark_csv():
    landmarks = ['class']
    for i in range(1,469):
        landmarks += ['x{}'.format(i),'y{}'.format(i),'z{}'.format(i),'v{}'.format(i)]
    # print(len(landmarks))
    return pd.DataFrame(columns=landmarks), landmarks


# capture frames
# 
def capture_frames( target, n_frames = 50):
    
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
            # mp_drawing_utils.draw_landmarks(image,result.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
            # mp_drawing_utils.draw_landmarks(image,result.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
            # mp_drawing_utils.draw_landmarks(image,result.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
            # #image = cv2.resize(image, (1000,7000))
            
            try:
                coord1=tuple(np.multiply(np.array((result.face_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x,result.face_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y)),[640,480]).astype('int'))
                
                z = result.face_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].z
                # cv2.putText(img=image, text = str(z)[:8], org = np.array(coord1)+np.array((100,-110)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=3)
                
                coord1=np.array(coord1)+np.array((100,-110))
                coord2=tuple(np.multiply(np.array((result.face_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,result.face_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),[640,480]).astype('int'))
                
                coord2=np.array(coord2)+np.array((-60,130))
                
                z = result.face_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].z
            
                if z<-0.025 and z>-0.032:
                    image=cv2.rectangle(image,coord1,coord2,(0,255,0),5)
                    
                    face=np.array(result.face_landmarks.landmark)
                    face_row=list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in face]).flatten())
                    face_row.insert(0, target)
                    added=pd.DataFrame([face_row],columns=landmarks)
                    work=pd.concat([work,added])
                    track_frames += 1
                
                else:
                    image=cv2.rectangle(image,coord1,coord2,(0,0,255),5)
                cv2.imshow('raw',image)    
            except:
                cv2.imshow('raw',image)
            
            if (cv2.waitKey(10) & 0xFF == ord('Q')) or track_frames > n_frames :
                break
    # destroy frame window
    cap.release()
    cv2.destroyAllWindows()
    
    return work


