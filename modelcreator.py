# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:15:45 2022

@author: ASUS
"""

from sklearn.pipeline  import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

#%%
pipeline={
    'rl':make_pipeline(StandardScaler(),LogisticRegression()),
     'rc':make_pipeline(StandardScaler(),RidgeClassifier()),
     'rf':make_pipeline(StandardScaler(),RandomForestClassifier()),
     'gb':make_pipeline(StandardScaler(),GradientBoostingClassifier())}

#%%

print(pipeline.keys())

#%%

print(list(pipeline.values())[0])
#%%
face_data=pd.read_csv('face_data3.csv')

print(face_data.head())
#%%
X=face_data.drop('class',axis=1)
y=face_data['class']
#%%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1234)

#%%
fit_model={}

for algo,pipeline in pipeline.items():
    model=pipeline.fit(X_train,y_train)
    fit_model[algo]=model
    #%%
for algo,model in fit_model.items():
    prediction=model.predict(X_test)
    score=accuracy_score(y_test,prediction)
    print(algo,score)
#%%
    
with open('face4.pkl','wb') as f:
    pickle.dump(fit_model['gb'],f)















    