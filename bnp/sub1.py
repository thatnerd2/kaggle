
import pandas as pd
import numpy as np
import scipy as sci
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

is_numeric = train[:5].applymap(np.isreal).all(0) 
predictors = [col for col in train.columns[2:] if is_numeric[col]]

train = train.fillna(train.median())
test = test.fillna(test.median())
#logistic regression
alg = RandomForestClassifier()
alg.fit(train[predictors],train["target"])
predictions = alg.predict_proba(test[predictors])[:,0]

#save submission
submission = pd.DataFrame({ "PredictedProb": predictions,"ID":test.ID})
submission.to_csv('submission_bnp_example',index=False)