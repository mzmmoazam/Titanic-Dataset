import csv
import numpy as np

### I had only used numpy , as this is when I had just dived in
### this universe

### So , for beginners go search for pandas you will reduce the lines of code and it
### is awesome !!


### read data
with open('../Data/train.csv') as f:
    reader = csv.reader(f, delimiter=',')
    data = []
    for row in reader:
        data.append(row)

### labels
data_headers = data[0]

### get some fields only
for i in ["Name", "PassengerId", "Survived", "Ticket", "Fare", "Cabin"]:
    data_headers.remove(i)

### preprocessing and encoding

data = np.array(data[1:])
data = np.delete(data, [0, 3], 1)
order = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
data[data == ''] = '01111110'
train_result = data[:, 0]
data = np.delete(data, [0, 6], 1)
data = np.delete(data, 5, 1)
data = np.delete(data, 5, 1)
print(data_headers, data[0])
data[data == "male"] = 0
data[data == "female"] = 1
data[data == "S"] = 1
data[data == "Q"] = 0
data[data == "C"] = 2

### using various classifiers

# from sklearn.naive_bayes import GaussianNB
# clf=GaussianNB()
# from sklearn.tree import DecisionTreeClassifier
# clf=DecisionTreeClassifier()
# from sklearn.svm import SVC
# clf=SVC()
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2)
# clf=AdaBoostClassifier()
# from sklearn.neighbors import KNeighborsClassifier
# clf=KNeighborsClassifier()
# print(np.array(['','1']).astype(np.float),"jsbd")
# print(len(data.astype(np.float)),"############",len(train_result.astype(np.float)))


### fit data to classifier
clf.fit(data.astype(np.float), train_result.astype(np.float))

# Testing data
data = []
with open('../Data/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    data = []
    for row in reader:
        data.append(row)
# print(len(data))
data_headers = data[0]

### preprocessing for test data

for i in ["Name", "PassengerId", "Ticket", "Fare", "Cabin"]:
    data = np.delete(data, data_headers.index(i), 1)
    data_headers.remove(i)
data = np.array(data[1:])
data[data == ''] = '01111110'
data[data == "male"] = 0
data[data == "female"] = 1
data[data == "S"] = 1
data[data == "Q"] = 0
data[data == "C"] = 2
# print(len(data),len(order),data,"end data")
test_data = np.array(data[:, data_headers.index(order[0])])
for i in order[1:]:
    test_data = np.vstack((test_data, data[:, data_headers.index(i)]))
# print(data_headers,"jdbfue",test_data,"jdbueb")
with open('../Data/gender_submission.csv') as f:
    reader = csv.reader(f, delimiter=',')
    test_labels = []
    for row in reader:
        test_labels.append(row[1])
print(len(test_labels))
test_labels = np.array(test_labels[1:])

ans = clf.predict(test_data.astype(np.float).T)

ans1 = np.array([range(892, 1310)])

ans = np.vstack((ans1.astype(np.int), ans.astype(np.int))).T

np.savetxt("fo1o.csv", ans, delimiter=",", fmt='%d')

