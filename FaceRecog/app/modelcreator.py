

# imports
# 
from sklearn.pipeline  import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle, time, os, uuid
from .holistics import capture_frames, landmark_csv
from . import base_dir


pipeline={
    # 'rl':make_pipeline(StandardScaler(),LogisticRegression()),
    # 'rc':make_pipeline(StandardScaler(),RidgeClassifier()),
    # 'rf':make_pipeline(StandardScaler(),RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(),GradientBoostingClassifier())
}


# populate initial data
# 
def populate_initial_data(n=468*4):
    arr = np.array([['unknown', *[0 for i in range(468*4)]]])
    populated_data = pd.DataFrame(data=arr, columns=landmark_csv()[1])
    populated_data.to_csv(base_dir / "data/face_data.csv", index=False)


# read training data
# 
def read_training_data():
    try:
        data = pd.read_csv(base_dir / 'data/face_data.csv')
        return data
    except FileNotFoundError:
        populate_initial_data()
        return pd.read_csv(base_dir / 'data/face_data.csv')


# update model training data
# 
def update_data(data: pd.DataFrame):
    # updating process
    _t = time.strftime("%d-%m-%Y-%H-%M", time.localtime())
    _uuid = str(uuid.uuid4())
    if os.path.exists(base_dir / "data/face_data.csv"):
        os.rename(base_dir / "data/face_data.csv", base_dir / f"data/face_data-{_uuid}-{_t}.csv")
    else:
        populate_initial_data()
        
    data.to_csv(base_dir / "data/face_data.csv", index=False)


# create model
# 
def create_model(target, algorithm='gb'):
    # capturing data
    captured_data = capture_frames(target)
    prev_data = read_training_data()
    #  update the existing model data
    update_data(pd.concat([prev_data, captured_data]))
    face_data = read_training_data()
    
    X=face_data.drop('class',axis=1)
    y=face_data['class']
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
    
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

