#%%-----------------------------------------------------------------------
# Exercise
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
# 1- Import the wine_data.csv file. Add the column names into the data frame.

# names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium",
#         "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins",
#         "Color_intensity", "Hue", "OD280", "Proline"]
#%%-----------------------------------------------------------------------
#Sol
import pandas as pd
names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium",
         "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins",
         "Color_intensity", "Hue", "OD280", "Proline"]
df = pd.read_csv('wine_data.csv', names = names)



#%%-----------------------------------------------------------------------
# 2- Check the dataset and explain the features using statistical measures.
#%%-----------------------------------------------------------------------
#Sol
print(df.head())

print(df.describe())



#%%-----------------------------------------------------------------------
# 3- Find the target feature and then create training and testing set.
#%%-----------------------------------------------------------------------
#Sol
from sklearn.model_selection import train_test_split
X = df.values[:,1:]
y = df.values[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


#%%-----------------------------------------------------------------------
# 4- You need to apply pre processsing on the dataset and explain what method did you use.
#%%-----------------------------------------------------------------------
#Sol
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.fit_transform(X_test)
"""
Imported the standard scalar to scale the features to allow easier processing
of the data for the model.
"""
#%%-----------------------------------------------------------------------
# 5- Train a NN model for the wine dataset.
#%%-----------------------------------------------------------------------
#Sol
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train.ravel())

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


#%%-----------------------------------------------------------------------
# 6- Change the following parameters and explain the changes.
# i. Use sigmoid transfer function.
# ii. Use Early stopping.
# iii. Use 2, 3 and 4 layer with any number of neurons below 15.
# iv. Use momentum with the values of 0.95.
# v. Use SGD as solver.
#%%-----------------------------------------------------------------------
#Sol
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation = 'logistic', max_iter=1000)
mlp.fit(X_train, y_train.ravel())

predictions = mlp.predict(X_test)

print('Sigmoid change')
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, early_stopping = True)
mlp.fit(X_train, y_train.ravel())

predictions = mlp.predict(X_test)

print('Early Stopping change')
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

mlp = MLPClassifier(hidden_layer_sizes=(6, 7, 8), max_iter=1000)
mlp.fit(X_train, y_train.ravel())

predictions = mlp.predict(X_test)

print('Neuron change')
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, momentum = 0.95)
mlp.fit(X_train, y_train.ravel())

predictions = mlp.predict(X_test)

print('Momentum change')
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, solver = 'sgd')
mlp.fit(X_train, y_train.ravel())

predictions = mlp.predict(X_test)

print('SGD Solver change')
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#%%-----------------------------------------------------------------------
# 7- For each changes of parameters you have done calculate the confusion matrix and classification report.
# Explain the results.
#%%-----------------------------------------------------------------------
#Sol

'''
Sigmoid change: Accuracy score decreased, meaning the model performed worse than the base model. It could not classify class 1 at all.
Early Stopping change: Accuracy score decreased, meaning the model performed worse than the base model. It could not classify class 1 or class 3 at all.
Neuron change: Accuracy score decreased a bit, meaning it performed slightly worse than the base model.
Momentum change: Accuracy score decreased, meaning the model performed worse than the base model. It could not classify class 3 at all.
SGD Solver change: Accuracy score decreased, meaning the model performed worse than the base model. It could not classify class 3 at all.

'''




#%%-----------------------------------------------------------------------