

# imports
# 
from email.mime import base
from threading import local
from sklearn.pipeline  import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle, time, os
from .holistics import capture_frames
from . import base_dir


pipeline={
    'rl':make_pipeline(StandardScaler(),LogisticRegression()),
    'rc':make_pipeline(StandardScaler(),RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(),RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(),GradientBoostingClassifier())
}


# read training data
# 
def read_training_data():
    return pd.read_csv(base_dir / 'data/face_data.csv')


# update model training data
# 
def update_data(data: pd.DataFrame):
    # updating process
    _t = time.strftime("%d-%m-%Y-%H-%M", time.localtime())
    if os.path.exists(base_dir / "data/face_data.csv"):
        os.rename(base_dir / "data/face_data.csv", base_dir / f"data/face_data-{_t}.csv")
    data.to_csv(base_dir / "data/face_data.csv")


# create model
# 
def create_model(target, algorithm='gb'):
    # capturing data
    captured_data = capture_frames(target)
    #  update the existing model data
    update_data(captured_data)
    face_data = read_training_data()
    
    X=face_data.drop('class',axis=1)
    y=face_data['class']
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1234)
    
    fit_model={}

    for algo, _pipeline in pipeline.items():
        model = _pipeline.fit(X_train, y_train)
        fit_model[algo]=model
        print(f"{algo} algorithm is done!")
        
    algo_scores = {}
    for algo,model in fit_model.items():
        prediction=model.predict(X_test)
        score=accuracy_score(y_test,prediction)
        algo_scores[algo] = score
        
    with open(base_dir / 'data/face_model.pkl','wb') as f:
        pickle.dump(fit_model[algorithm],f)
    
    return algorithm, score

