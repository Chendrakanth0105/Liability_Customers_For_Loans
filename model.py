import numpy as np
import pandas as pd
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
data = pd.read_csv('input.csv')
#print(data.head())
data = data.drop(['Experience','ID','ZIP Code'], axis=1)
X = data.drop('Personal Loan', axis=1).values
Y = data['Personal Loan'].values
#print(data.head())
#print(data.columns)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)


knn = KNeighborsClassifier(n_neighbors=3 , weights = 'uniform', metric='euclidean')
print(knn.fit(X_train, y_train))


pickle.dump(knn,open('model.pkl','wb'))
