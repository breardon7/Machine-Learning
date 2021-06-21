#%%-----------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

#%%-----------------------------------------------------------------------
# Read in dataset
#%%-----------------------------------------------------------------------

df = pd.read_csv('Train.csv')
df = df.sample(frac=0.01, random_state = 1)
print(df.head())
print(df.shape)
'''print(df.columns)
print(df.dtypes)
print(df.info)'''

#%%-----------------------------------------------------------------------
# Preprocessing
#%%-----------------------------------------------------------------------
# Move Target to first column
target = 'Accident_Severity'
first_col = df.pop(target)
df.insert(0, target,  first_col)
df = df.replace({'Accident_Severity': {1:0, 2:0, 3:1}})
print(df.head())
# Check NA value count
############ Check to see if null values are previously filled with zeroes or another value ############
null_values = df.isnull().sum().sort_values(ascending=False)
#print(null_values[null_values > 0])

# Drop Values with too many missing values
drop_cols = ['Junction_Detail']
df = df.drop(labels=drop_cols, axis = 1)
print(df.shape)

# Fill Missing Values (Simple Imputing)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df = pd.DataFrame(imp_mode.fit_transform(df), columns=df.columns)
#print(df.isnull().sum())


# Transform categorical features
le = LabelEncoder()
df['Accident_Index'] = le.fit_transform(df['Accident_Index'])
df['Date'] = le.fit_transform(df['Date'])
df['Time'] = le.fit_transform(df['Time'])
df['Local_Authority_(Highway)'] = le.fit_transform(df['Local_Authority_(Highway)'])
df['Road_Type'] = le.fit_transform(df['Road_Type'])
df['Junction_Control'] = le.fit_transform(df['Junction_Control'])
df['Pedestrian_Crossing-Human_Control'] = le.fit_transform(df['Pedestrian_Crossing-Human_Control'])
df['Pedestrian_Crossing-Physical_Facilities'] = le.fit_transform(df['Pedestrian_Crossing-Physical_Facilities'])
df['Light_Conditions'] = le.fit_transform(df['Light_Conditions'])
df['Weather_Conditions'] = le.fit_transform(df['Weather_Conditions'])
df['Road_Surface_Conditions'] = le.fit_transform(df['Road_Surface_Conditions'])
df['Special_Conditions_at_Site'] = le.fit_transform(df['Special_Conditions_at_Site'])
df['Carriageway_Hazards'] = le.fit_transform(df['Carriageway_Hazards'])
df['Did_Police_Officer_Attend_Scene_of_Accident'] = le.fit_transform(df['Did_Police_Officer_Attend_Scene_of_Accident'])
df['LSOA_of_Accident_Location'] = le.fit_transform(df['LSOA_of_Accident_Location'])

# New Data
df_predict = pd.read_csv('Test_submission_netid_Ver_X.csv')

# Move Target to first column
target = 'Accident_Severity'
first_col = df_predict.pop(target)
df_predict.insert(0, target,  first_col)
df_predict = df_predict.drop(columns=['Junction_Detail'], axis = 1)



df_predict['Accident_Index'] = le.fit_transform(df_predict['Accident_Index'])
df_predict['Date'] = le.fit_transform(df_predict['Date'])
df_predict['Time'] = le.fit_transform(df_predict['Time'])
df_predict['Local_Authority_(Highway)'] = le.fit_transform(df_predict['Local_Authority_(Highway)'])
df_predict['Road_Type'] = le.fit_transform(df_predict['Road_Type'])
df_predict['Junction_Control'] = le.fit_transform(df_predict['Junction_Control'])
df_predict['Pedestrian_Crossing-Human_Control'] = le.fit_transform(df_predict['Pedestrian_Crossing-Human_Control'])
df_predict['Pedestrian_Crossing-Physical_Facilities'] = le.fit_transform(df_predict['Pedestrian_Crossing-Physical_Facilities'])
df_predict['Light_Conditions'] = le.fit_transform(df_predict['Light_Conditions'])
df_predict['Weather_Conditions'] = le.fit_transform(df_predict['Weather_Conditions'])
df_predict['Road_Surface_Conditions'] = le.fit_transform(df_predict['Road_Surface_Conditions'])
df_predict['Special_Conditions_at_Site'] = le.fit_transform(df_predict['Special_Conditions_at_Site'])
df_predict['Carriageway_Hazards'] = le.fit_transform(df_predict['Carriageway_Hazards'])
df_predict['Did_Police_Officer_Attend_Scene_of_Accident'] = le.fit_transform(df_predict['Did_Police_Officer_Attend_Scene_of_Accident'])
df_predict['LSOA_of_Accident_Location'] = le.fit_transform(df_predict['LSOA_of_Accident_Location'])

#df_predict = pd.DataFrame(imp_mode.fit_transform(df_predict), columns=df_predict.columns)


# Create train and test sets
# Test Data
X = df.values[:, 1:]
y = df.values[:, 0]
y=y.astype('int')
#print(np.unique(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Feature Scaling
sc = StandardScaler()

X_scaled = sc.fit_transform(X)

sc.fit(X_train)
X_train_scaled = sc.transform(X_train)
X_test_scaled = sc.transform(X_test)

# ------------------------------------------------------------------------------
# Gradient Boosting

# -------------------------------------------------------------------------------------
# Create network
#mlp = MLPClassifier(max_iter=10000000)

'''
# Hyper-parameter space
parameter_space = {
    'hidden_layer_sizes': [(60,100,60)],
    'activation': ['identity', 'relu', 'logistic', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001],
    'learning_rate': ['constant','adaptive', 'invscaling'],
}

parameter_space = {
    'hidden_layer_sizes': [(60,100,60), (60,60,60), (70,20,70), (50,50,50)],
    'activation': ['tanh'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate': ['invscaling'],
}

# Run Gridsearch
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)

# Gridsearch Results

clf.fit(X_train, y_train)
x_predictions = clf.predict(X_test)
print(classification_report(y_test, x_predictions))

# Best parameter set
print('Best parameters found:\n', clf.best_params_)
'''

#Best Parameters Network
mlp = MLPClassifier(hidden_layer_sizes=(60, 100, 60), max_iter=10000, learning_rate='invscaling', solver="adam",
                            activation='tanh', alpha= 0.0001)

# MLPClassifier Results

mlp.fit(X_train, y_train)
x_predictions = mlp.predict(X_test)
print(classification_report(y_test, x_predictions))




