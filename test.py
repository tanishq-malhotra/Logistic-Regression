import numpy as np
import pandas as pd
import seaborn as sns
from LogisticRegression import LogisticRegression 

# reading the dataset
data = pd.read_csv('titanic_train.csv')

# dropping the unwanted features
data.drop('PassengerId', axis=1, inplace=True)
data.drop('Name', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)


# function to correct the age
def correctAge(col):
    age = col[0]
    pclass = col[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age

# apply the correctage functoin
data['Age'] = data[['Age','Pclass']].apply(correctAge, axis=1)
data.drop('Ticket', axis=1, inplace=True)

# changing the categorical column to separate
dummySex = pd.get_dummies(data['Sex'])
dummyEmbarked = pd.get_dummies(data['Embarked'])
data.drop(['Sex','Embarked'], axis=1, inplace=True)
# concatinating the new columns
data = pd.concat([data,dummySex,dummyEmbarked],axis=1)

# separating the features
X = data.drop('Survived',axis=1)
y = data['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

# model called if 1100 iterations
lg = LogisticRegression(iterations=1100)

lg.fit(X_train,y_train)

pred = lg.predict(X_test)

# for checking the accuracy
from sklearn.metrics import accuracy_score

print('accuracy of the model: {}'.format(accuracy_score(y_test,pred)*100))