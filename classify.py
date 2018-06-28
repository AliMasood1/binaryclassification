import keras
import numpy
import pandas
from keras import models
from keras.legacy import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 7


dataframe = pandas.read_csv("/home/alimasood/Documents/Deep_Learning_Project_One/Deep_Learning_Project_One/sonar.csv",
                            header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]
dataset = dataframe.values
le = LabelEncoder()
encoder= le.fit(Y)
inta = encoder.transform(Y)




#Starting
def baseline():

   numpy.random.seed(seed)
   #model_simple
   def create_baseline():
     network = Sequential()
     network.add(Dense(60, input_dim=60, activation='relu'))
     network.add(Dense(1, activation='sigmoid'))
     network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     return network

   estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
   kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
   results = cross_val_score(estimator, X, inta, cv=kfold)
   print(" without any effort  Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

   #"Results: 84.18% (6.44%)"

#standardized

def standardized():
    numpy.random.seed(seed)
    def create_baseline():
        network = Sequential()
        network.add(Dense(60, input_dim=60, activation='relu'))
        network.add(Dense(1, activation='sigmoid'))
        network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return network

    estimator=[]
    estimator.append(('standardize',StandardScaler()))
    estimator.append(('mlp',KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
    kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
    pipeline=Pipeline(estimator)
    results=cross_val_score(pipeline,X,inta,cv=kfold)
    print("standardized  Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    #"Results: 85.59% (7.46%)"

#smaller_model
def Smaller():
    numpy.random.seed(seed)
    def create_smaller():
      network = Sequential()
      network.add(Dense(30, input_dim=60, activation='relu'))
      network.add(Dense(1, activation='sigmoid'))
      network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
      return network


    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(pipeline, X, inta, cv=kfold)
    print("Smaller: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    # "85.06% (7.61%)"

#larger_model
def larger():
   numpy.random.seed(seed)
   def create_larger():
    network = Sequential()
    network.add(Dense(60, input_dim=60, activation='relu'))
    network.add(Dense(30, activation='relu'))
    network.add(Dense(1, activation='sigmoid'))
    network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return network

   estimators = []
   estimators.append(('standardize', StandardScaler()))
   estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=5, verbose=0)))
   pipeline = Pipeline(estimators)
   kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
   results = cross_val_score(pipeline, X, inta, cv=kfold)
   print("Larger: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
   #Larger: 84.61% (6.36%)

#mixture

def mixture():
    numpy.random.seed(seed)

    def create_baseline():
        network = Sequential()
        network.add(Dense(30, input_dim=60, activation='relu'))
        network.add(Dense(1, activation='sigmoid'))
        network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return network

    estimator = []
    estimator.append(('standardize', StandardScaler()))
    estimator.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=90, batch_size=5, verbose=0)))
    kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=seed)
    pipeline = Pipeline(estimator)
    results = cross_val_score(pipeline, X, inta, cv=kfold)
    print("standardized  Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    #Results: 87.87% (9.51%):)





mixture()