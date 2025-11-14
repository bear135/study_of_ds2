import numpy as np 
import pandas as pd 

'''
q1. Pima Indians Diabetes (당뇨병 여부 판단) 
- 예측컬럼: Outcome (0 정상, 1 당뇨) 당뇨병일 확률 예측
- 평가지표: roc-auc
- 제출파일명: result.csv (1개컬럼, 컬럼명 pred)
'''
train = pd.read_csv('data/diabetes_train.csv')
test = pd.read_csv('data/diabetes_test.csv')
train.shape, test.shape 
train.head()

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

y = train['Outcome']
train = train.drop('Outcome',axis=1)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score 
model = RandomForestClassifier(random_state=120)
model.fit(X_train, y_train)

model_pred = model.predict(X_valid)
score = roc_auc_score(y_valid, model_pred)
score    # 0.6858974358974359

pred = model.predict_proba(test)
result = pd.DataFrame({
    'pred':pred[:,1]
})
result.to_csv('result.csv',index=False)

'''
q2. 학업위험 예측 (Classification with an Academic Success Dataset)
- 다중분류, Accuracy
'''
train = pd.read_csv('data/academic_success/train.csv')
test = pd.read_csv('data/academic_success/test.csv')
train.shape, test.shape 
train.head()
train.info()

y = train['Target']
train = train.drop(['id','Target'],axis=1)
test = test.drop('id',axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=120)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score 
model_pred = model.predict(X_valid)
score = accuracy_score(y_valid, model_pred)
score   # 0.829064296915839

pred = model.predict(test)
result = pd.DataFrame({'pred':pred})
result.to_csv('result.csv',index=False)

