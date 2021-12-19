import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree

#read data
data_train = pd.read_csv("/Users/weiranlyu/Desktop/2021 Fall/CS6350/project/ilp2021f/train_final.csv")
data_test = pd.read_csv("/Users/weiranlyu/Desktop/2021 Fall/CS6350/project/ilp2021f/test_final.csv")

#data cleanup
ID = data_test. iloc[:, 0]
data_test = data_test.iloc[: , 1:]
#treat missing value "?" as the majority of the other values of the same attribute
data_train['workclass'] = data_train['workclass'].replace(['?'], data_train['workclass'].mode())
data_train['occupation'] = data_train['occupation'].replace(['?'], data_train['occupation'].mode())
data_train['native.country'] = data_train['native.country'].replace(['?'], data_train['native.country'].mode())

data_test['workclass'] = data_test['workclass'].replace(['?'], data_test['workclass'].mode())
data_test['occupation'] = data_test['occupation'].replace(['?'], data_test['occupation'].mode())
data_test['native.country'] = data_test['native.country'].replace(['?'], data_test['native.country'].mode())

#convert categorical values to numerical values
def sex_to_numeric(x):
    if x =='Male':
        return 0
    else:
        return 1

data_train["sex"] = data_train["sex"].apply(sex_to_numeric)
data_test["sex"] = data_test["sex"].apply(sex_to_numeric)

cleanup_strs = {"workclass":     {"Private": 0, "Self-emp-not-inc": 1,"Self-emp-inc": 2,"Federal-gov": 3,"Local-gov": 4,"State-gov": 5, "Without-pay": 6,"Never-worked": 7},
                "education": {"Bachelors": 0, "Some-college": 1, "11th": 2, "HS-grad": 3,
                                  "Prof-school": 4, "Assoc-acdm": 5, "Assoc-voc":6, "9th": 7, "7th-8th":8, "12th": 9, "Masters":10, "1st-4th": 11, "10th":12, "Doctorate": 13, "5th-6th":14, "Preschool":15},
               "marital.status":     {"Married-civ-spouse":0, "Divorced":1, "Never-married":2, "Separated":3, "Widowed":4, "Married-spouse-absent":5, "Married-AF-spouse":6},
                "occupation": {"Tech-support":0, "Craft-repair":1, "Other-service":2, "Sales":3, "Exec-managerial":4, "Prof-specialty":5, "Handlers-cleaners":6, 
"Machine-op-inspct":7, "Adm-clerical":8, "Farming-fishing":9, "Transport-moving":10, "Priv-house-serv":11, "Protective-serv":12, "Armed-Forces":13},
               "relationship":     {"Wife":0, "Own-child":1, "Husband":2, "Not-in-family":3, "Other-relative":4, "Unmarried":5},
                "race": {"White":0, "Asian-Pac-Islander":1, "Amer-Indian-Eskimo":2, "Other":3, "Black":4},
               "native.country":     {"United-States":0, "Cambodia":1, "England":2, "Puerto-Rico":3, "Canada":4, "Germany":5, "Outlying-US(Guam-USVI-etc)":6, "India":7, "Japan":8, "Greece":9, 
"South":10, "China":11, "Cuba":12, "Iran":13, "Honduras":14, "Philippines":15, "Italy":16, "Poland":17, "Jamaica":18, "Vietnam":19, "Mexico":20, "Portugal":21, "Ireland":22, 
"France":23, "Dominican-Republic":24, "Laos":25, "Ecuador":26, "Taiwan":27, "Haiti":28, "Columbia":29, "Hungary":30, "Guatemala":31, "Nicaragua":32, "Scotland":33, 
"Thailand":34, "Yugoslavia":35, "El-Salvador":36, "Trinadad&Tobago":37, "Peru":38, "Hong":39, "Holand-Netherlands":40}}

cleaned_train = data_train.replace(cleanup_strs)
cleaned_test = data_test.replace(cleanup_strs)

X_train, Y_train = cleaned_train.iloc[:,:-1], cleaned_train.iloc[:,-1]
ID = ID.to_numpy()

#SVM
model_svm = svm.SVC(kernel='rbf',C=20,gamma='scale')
model_svm.fit(X_train, Y_train)
prediction_svm = model_svm.predict(cleaned_test)
print(prediction_svm)

output = {"ID": ID, "Prediction": prediction_svm}
output = pd.DataFrame(output)
output.to_csv('/Users/weiranlyu/Desktop/prediction_svm.csv')

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, Y_train)
prediction_svc = clf.predict(cleaned_test)

output = {"ID": ID, "Prediction": prediction_sgdc}
output = pd.DataFrame(output)
output.to_csv('/Users/weiranlyu/Desktop/prediction_sgdc.csv')

#Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)
prediction_lr = model_lr.predict(cleaned_test)
print(prediction_lr)

output = {"ID": ID, "Prediction": prediction_lr}
output = pd.DataFrame(output)
output.to_csv('/Users/weiranlyu/Desktop/prediction_lr.csv')

#K-NN 
model_knn = KNeighborsClassifier(n_neighbors=300)
model_knn.fit(X_train, Y_train)
prediction_knn = model_knn.predict(cleaned_test)
print(prediction_knn)

output = {"ID": ID, "Prediction": prediction_knn}
output = pd.DataFrame(output)
output.to_csv('/Users/weiranlyu/Desktop/prediction_knn.csv')

#Decision tree
decisionTree = tree.DecisionTreeClassifier()
model_dt = decisionTree.fit(X_train, Y_train)
prediction_dt = model_dt.predict(cleaned_test)
print(prediction_dt)

output = {"ID": ID, "Prediction": prediction_dt}
output = pd.DataFrame(output)
output.to_csv('/Users/weiranlyu/Desktop/prediction_dt.csv')

#Adaboost
abc = AdaBoostClassifier(n_estimators=500, learning_rate=1, random_state=0)
model_ada = abc.fit(X_train, Y_train)
prediction_ada = model_ada.predict(cleaned_test)
print(prediction_ada)

output = {"ID": ID, "Prediction": prediction_ada}
output = pd.DataFrame(output)
output.to_csv('/Users/weiranlyu/Desktop/prediction_ada.csv')

model_rf = RandomForestClassifier(random_state=0)
model_rf.fit(X_train, Y_train)
prediction_rf = model_rf.predict(cleaned_test)
print(prediction_rf)

output = {"ID": ID, "Prediction": prediction_rf}
output = pd.DataFrame(output)
output.to_csv('/Users/weiranlyu/Desktop/prediction_rf.csv')