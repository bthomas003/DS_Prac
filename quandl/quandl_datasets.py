# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url1 = "https://www.quandl.com/api/v3/datasets/EOD/V.csv"
url2 = "https://www.quandl.com/api/v3/datasets/EOD/KO.csv"
url3 = "https://www.quandl.com/api/v3/datasets/EOD/WMT.csv"
url4 = "https://www.quandl.com/api/v3/datasets/EOD/GS.csv"
url5 = "https://www.quandl.com/api/v3/datasets/EOD/BA.csv"

#Filter data and add name column
df1 = pandas.read_csv(url1, usecols=[8,9,10,11])
df1['name'] = 'Visa'
df2 = pandas.read_csv(url1, usecols=[8,9,10,11])
df2['name'] = 'Coke'
df3 = pandas.read_csv(url1, usecols=[8,9,10,11])
df3['name'] = 'WalMart'
df4 = pandas.read_csv(url1, usecols=[8,9,10,11])
df4['name'] = 'GoldSachs'
df5 = pandas.read_csv(url1, usecols=[8,9,10,11])
df5['name'] = 'Boeing'

#Combine all datasets into one
dc0 = df1.append(df2)
dc1 = dc0.append(df3)
dc2 = dc1.append(df4)
dc3 = dc2.append(df5)
dataset = dc3

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('name').size())

# Split-out validation dataset
array = dataset.values
X = array[:,0:3]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

