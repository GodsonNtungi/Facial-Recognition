# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:39:03 2022

@author: ASUS
"""

import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import pickle
import pathlib

base_dir = pathlib.Path(__file__).parent
# In[9]:
with open(base_dir / 'FaceRecog/data/face_model.pkl','rb') as f:
    model=pickle.load(f)
#%%
print(model)

#%%
mp_drawing_utils=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles
print('done')
#%%

cap=cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        log,frame=cap.read()
        
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=holistic.process(image)
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        #mp_drawing_utils.draw_landmarks(
        #image,
        #result.face_landmarks,
        #mp_holistic.FACEMESH_TESSELATION,
        #landmark_drawing_spec=None,
        #connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        #mp_drawing_utils.draw_landmarks(image,result.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
        #mp_drawing_utils.draw_landmarks(image,result.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        #mp_drawing_utils.draw_landmarks(image,result.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        #image = cv2.resize(image, (1000,7000))
        #cv2.imshow('raw',image)
        try:
            coord1=tuple(np.multiply(np.array((result.face_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x,result.face_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y)),[640,480]).astype('int'))
            print(coord1)
            coord1=np.array(coord1)+np.array((80,-110))
            coord2=tuple(np.multiply(np.array((result.face_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,result.face_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),[640,480]).astype('int'))
            
            coord2=np.array(coord2)+np.array((-40,100))
            print(coord2)
            image=cv2.rectangle(image,coord1,coord2,(0,255,0),5)
            cv2.imshow('raw',image)
            
            
            face=np.array(result.face_landmarks.landmark)
            face_row=list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in face]).flatten())
            added=pd.DataFrame([face_row])
            prediction=model.predict(added)[0]
            prediction_prob=model.predict_proba(added)[0][0]
            
            print(prediction,prediction_prob)
            
        except:
            cv2.imshow('raw',image)
        
        
        
        if cv2.waitKey(10) & 0xFF == ord('Q') :
            break
cap.release()
cv2.destroyAllWindows()

            
            
            
            
            
            
            
            
            
            
            