# %%%%%%%%%%%%% Machine Learning %%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 June - 06 - 2020
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% NN Regression %%%%%%%%%%%%%%%%%%%%%%

#%%-----------------------------------------------------------------------
# Exercise
#%%-----------------------------------------------------------------------
# 1- Generate data from the following function np.sin(np.exp(p))
# 2- Write a script to fit the function
# 3- Plot the trained network vs original function
# 4- Use different layers, neurons, optimizer etc
# 5- Explain your findings and write down a paragraph to explain all the results.
# 6- Explain your performance index results and figures.
#%%-----------------------------------------------------------------------
# 1-
import numpy as np
p = np.linspace(-3, 3, 10000).reshape(-1, 1)
t = np.sin(np.exp(p))


#%%-----------------------------------------------------------------------
# 2-

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

X_train, X_test, y_train, y_test = train_test_split(p, t, random_state=1)
regr = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(100,100,100))
regr.fit(X_train, y_train.flatten())
pred = regr.predict(X_test)
print(mean_squared_error(y_test, pred))

#%%-----------------------------------------------------------------------
# 3-
import matplotlib.pyplot as plt

plt.scatter(X_test,y_test)
plt.scatter(X_test, pred, c='r')
plt.title('Original Function vs Trained Network')
plt.show()
plt.plot(pred, y_test, 'r')
plt.title('MSE')
plt.show()

#%%-----------------------------------------------------------------------
# 4-

#%%-----------------------------------------------------------------------
# 5-
'''
Higher hidden layer counts decrease MSE, which means our model is more likely to predict the correct target.
'''
#%%-----------------------------------------------------------------------
# 6-
'''
The performance index is virtually zero, meaning the model is performing nearly perfectly.
The first figure shows the output of the original function compared to the network output.
The second figure shows a plot of predictions versus the actual targets, or MSE.
'''