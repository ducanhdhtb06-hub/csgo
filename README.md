import pandas as pd
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LinearRegression

from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error ,r2_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
data = pd.read_csv("StudentScore.xls")
x = data.drop(["math score"],axis = 1)
y = data["math score"]
x_train , x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
num_tranformer = Pipeline(steps = [
    ("imputer" , SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler())
])
nom_tranformer = Pipeline(steps = [
    ("imputer" , SimpleImputer(strategy = "most_frequent")) ,
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
student_values = ["some high school" ,"high school" , "some college" , "associate's degree","bachelor's degree","master's degree" ]
gender_value = x["gender"].unique()
lunch_value = x["lunch"].unique()
test_value = x["test preparation course"].unique()

ord_transformer = Pipeline(steps = [
    ("imputer" , SimpleImputer(strategy = "most_frequent")),
    ("ord" , OrdinalEncoder(categories = [student_values, gender_value , lunch_value, test_value]))
])
preprocessor = ColumnTransformer(transformers = [
    ("num_features" ,num_tranformer ,["reading score" , "writing score"] ),
    ("ord_features" , ord_transformer ,["parental level of education","gender" ,"lunch" , "test preparation course"] ),
    ("nom_features" , nom_tranformer ,["race/ethnicity"] ),
])
reg = Pipeline(steps = [
    ("preprocessor" , preprocessor),
    ("regressor" , LinearRegression() )
])
reg.fit(x_train , y_train)
y_pred = reg.predict(x_test)
for i , j in zip(y_pred , y_test):
    print(i," ", j)
print("MAE {}".format(mean_absolute_error(y_test, y_pred)))
print("MSE {}".format(mean_squared_error(y_test, y_pred)))
print("R2 {}".format(r2_score(y_test, y_pred)))
