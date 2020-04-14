import pandas as pd
import numpy as np

df_churn = pd.read_csv('https://raw.githubusercontent.com/rujual/telco_churn_pipeline/master/Data1.csv')  #('Data1.csv')
print('data read complete')

import pandas as pd
import numpy as np

print("one hot encoding op started")
empty_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

for i in empty_cols:
    df_churn[i]=df_churn[i].replace(" ",np.nan)

df_churn.drop(['customerID'], axis=1, inplace=True)
df_churn = df_churn.dropna()
binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling']

for i in binary_cols:
    df_churn[i] = df_churn[i].replace({"Yes":1,"No":0})

#Encoding column 'gender'
df_churn['gender'] = df_churn['gender'].replace({"Male":1,"Female":0})


category_cols = ['PaymentMethod','MultipleLines','InternetService','OnlineSecurity',
               'OnlineBackup','DeviceProtection',
               'TechSupport','StreamingTV','StreamingMovies','Contract']

for cc in category_cols:
    dummies = pd.get_dummies(df_churn[cc], drop_first=False)
    dummies = dummies.add_prefix("{}#".format(cc))
    df_churn.drop(cc, axis=1, inplace=True)
    df_churn = df_churn.join(dummies)

df_churn['Churn'] = df_churn['Churn'].replace({"Yes":1,"No":0})
print("Encoding Complete")
df_churn1 = df_churn.copy(deep=True)


import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import json

print("op started")

y1 = df_churn['Churn']
X1 = df_churn.drop(['Churn'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X1, y1, random_state=0)

#print(X_train.dtypes)
print("tt done")
#print(X_train.dtypes)
#print(y_train.dtype)

sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
X_test_res, y_test_res = sm.fit_sample(X_test, y_test)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [2,4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

rfc=RandomForestClassifier(random_state=42,n_estimators=100)
gsv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
rfc.fit(X_train_res, y_train_res)

rfc_best=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 50, max_depth=8,
                                criterion='gini')

rfc_best.fit(X_train, y_train) #(X_train_res, y_train_res)
X_test_res, y_test_res = sm.fit_sample(X_test, y_test)
y_test_pred = rfc_best.predict(X_test) #_res)
rf_score = rfc_best.score(X_test, y_test)  #(X_test_res, y_test_res)
conf = confusion_matrix(y_test, y_test_pred)
print('Rf Conf:',conf)


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb

df_churn = df_churn1

print("op started")
y1 = df_churn['Churn']
X1 = df_churn.drop(['Churn'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X1, y1, random_state=0)


sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
X_test_res, y_test_res = sm.fit_sample(X_test, y_test)

clfxg = xgb.XGBClassifier(objective='binary:logistic', verbosity=0, max_depth=2, eta = 1, silent=0)
clfxg.fit(X_train_res, y_train_res) #, num_round, watchlist)

y_test_pred = clfxg.predict(X_test_res)
conf = confusion_matrix(y_test_res, y_test_pred)
print("XGB Conf:", conf)

