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
# In[9]:
with open('mood.pkl','rb') as f:
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
        mp_drawing_utils.draw_landmarks(
        image,
        result.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        #mp_drawing_utils.draw_landmarks(image,result.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
        #mp_drawing_utils.draw_landmarks(image,result.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        #mp_drawing_utils.draw_landmarks(image,result.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        #image = cv2.resize(image, (1000,7000))
    
        cv2.imshow('raw',image)
        try:
            face=np.array(result.face_landmarks.landmark)
            face_row=list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in face]).flatten())
            added=pd.DataFrame([face_row])
            prediction=model.predict(added)[0]
            prediction_prob=model.predict_proba(added)[0][0]
            
            print(prediction,prediction_prob)
            
        except:
            pass
        
        
        
        if cv2.waitKey(10) & 0xFF == ord('Q') :
            break
cap.release()
cv2.destroyAllWindows()

            
            
            
            
            
            
            
            
            
            
            