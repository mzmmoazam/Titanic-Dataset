import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import tflearn

train = pd.read_csv('../Data/train.csv')
_test = pd.read_csv('../Data/test.csv')

print(train.head())
cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train = train[cols]

print(_test.head())
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
test = _test[cols]

# Check which columns have missing data in train data
for column in train.columns:
    if np.any(pd.isnull(train[column])) == True:
        print(column)

# so age and Embarked have missing data in train

# age --> median , for NaN values
train["Age"] = train["Age"].fillna(train["Age"].median())

# embarked --> most commmon data value

# we can find the most common value from here
print(train.Embarked.value_counts())

# So it s 'S'
train["Embarked"] = train["Embarked"].fillna('S')


# Check which columns have missing data in test data
for column in test.columns:
    if np.any(pd.isnull(test[column])) == True:
        print(column)

# Age and fare have missing data in test

# age --> median
test["Age"] = test["Age"].fillna(train["Age"].median())

# fare --> median
test["Fare"] = test["Fare"].fillna(train["Fare"].median())

# feature engineering

# Using siblings and parents we can give us the family size

for data in [train, test]:
    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

def filter_family_size(x):
    if x == 1:
        return '0'  #  alone
    elif x < 4:
        return '1'  #  small family
    else:
        return '2'  #  big family

for data in [train, test]:
    data['FamilySize'] = data['FamilySize'].apply(filter_family_size)

# one hot encoding of features
### Embarked
train.loc[train["Embarked"] == 'S', "Embarked"] = 0
train.loc[train["Embarked"] == 'C', "Embarked"] = 1
train.loc[train["Embarked"] == 'Q', "Embarked"] = 2

test.loc[test["Embarked"] == 'S', "Embarked"] = 0
test.loc[test["Embarked"] == 'C', "Embarked"] = 1
test.loc[test["Embarked"] == 'Q', "Embarked"] = 2

### Sex
train.loc[train["Sex"]=="male","Sex"]=0
train.loc[train["Sex"]=="female","Sex"]=1

test.loc[test["Sex"]=="male","Sex"]=0
test.loc[test["Sex"]=="female","Sex"]=1


# creating our model
def net(X,Y):
    net = input_data(shape=[None, 7], name='input')
    net = fully_connected(net,32,activation='relu', name='f_c_1')
    net = fully_connected(net,124,activation='relu', name='f_c_2')
    # net = fully_connected(net,124,activation='relu', name='f_c_3')
    # net = fully_connected(net,124,activation='relu', name='f_c_4')
    net = fully_connected(net,32,activation='relu', name='f_c_5')
    net = fully_connected(net,2,activation='softmax')
    net = regression(net, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=3)
    model.fit({'input': X}, {'target': Y}, n_epoch=25,
              # validation_set=({'input': x}, {'target': y}),
              # validation_set=0.3,
              snapshot_step=500, show_metric=True, run_id='tflearn_titanic')
    return model

print(train.head())
# train model
model = net(np.array(train[cols]),to_categorical(np.array(train['Survived']),2))
# test model
test_labels = [np.argmax(i)for i in model.predict(np.array(test[cols]))]

# store the predictions
pred = pd.DataFrame({
    "PassengerId" : _test["PassengerId"],
    "Survived" : test_labels
    })
pred.to_csv('pred.csv',index=False)
