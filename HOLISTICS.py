#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
# In[9]:


mp_drawing_utils=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles
print('done')


# In[27]:

landmarks=['class']
for i in range(1,469):
    landmarks += ['x{}'.format(i),'y{}'.format(i),'z{}'.format(i),'v{}'.format(i)]

print(len(landmarks))
work=pd.DataFrame(columns=landmarks)

#%%
cap=cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        log,frame=cap.read()
        
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=holistic.process(image)
        print(result.face_landmarks)
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
            face_row.insert(0,'sleepy')
            added=pd.DataFrame([face_row],columns=landmarks)
            work=pd.concat([work,added])
    
        except:
            pass
        
        
        
        if cv2.waitKey(10) & 0xFF == ord('Q') :
            break
cap.release()
cv2.destroyAllWindows()

#%%
print(work.index)
#%%
print(work.head())
#%%

work.to_csv('face_data1.csv',index=True)
#%%
print(work.iloc[0].compare(work.iloc[756]))

