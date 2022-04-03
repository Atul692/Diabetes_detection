import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data analysis
#Loading the diabetes dataset
diabetes_dataset = pd.read_csv(r"C:\Users\atulk\Downloads\diabetes (1).csv")
print(diabetes_dataset.head())
print(diabetes_dataset.shape)
print(diabetes_dataset['Outcome'].value_counts())

#separating the data and the labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

#Data standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(X)
# standardized_data = scaler.transform(X)

#Train test split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=7)
print(X.shape, X_train.shape, X_test.shape)

#training the model
classifier = svm.SVC(kernel='linear')

#traning the SVM classifier
classifier.fit(X_train,Y_train)

#Model eval by accuracy score
X_train_pred = classifier.predict(X_train)
training_acc = accuracy_score(X_train_pred,Y_train)

print("Accuracy of training data: ", training_acc)

X_test_pred = classifier.predict(X_test)
test_acc = accuracy_score(X_test_pred,Y_test)

print("Accuracy of test data: ", test_acc)

#Predting the outcome
# input_data = (1,89,66,23,94,28.1,0.167,21)
input_data = (0,137,40,35,168,43.1,2.288,33)
#Changing the input data into numpy array
input_data_asNP_array = np.asarray(input_data)

#reshape the array
input_data_reshaped = input_data_asNP_array.reshape(1,-1)

#standardize the input data
standard_data = scaler.transform(input_data_reshaped)
print(standard_data)


pred = classifier.predict(standard_data)
# print(pred)

if(pred[0]==0):
    print("The person is not diabetic!! You can eat sweets :)")
else:
    print("The person is having diabetes please avoid sweets.")
