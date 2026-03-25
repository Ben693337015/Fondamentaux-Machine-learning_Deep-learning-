# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:38:52 2026

@author: abdou
"""

import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

x,y = make_regression(n_samples=100,n_features=1, noise=10)

plt.scatter(x, y)

"il faut verifier la dimention des nos matrices"

print(x.shape)
y =  y.reshape(100,1)

print(y.shape)

# Matrice X
X = np.hstack((x,np.ones(x.shape)))
X

theta = np.random.randn(2,1)
theta

#2eme etape
#Modele lineare
def model(X,theta):
    return X.dot(theta)

plt.plot(x, model(X, theta), c='r')

# Fonction cout

def cost_fonction(X,y,theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) -y)**2)


cost_fonction(X, y, theta)

#Gradian et Descent de Gradiant

#definir la fonction pour le calcule de Gradiant

def gra(X,y,theta):
    m=len(y)
    return 1/m * X.T.dot(model(X,theta)-y)

#Algo descend de gradiant

def Descent_grad(X,y,theta,learning_rate,n_iteration):
    cost_history = np.zeros(n_iteration)
    
    
    for i in range(0,n_iteration):
        theta = theta - learning_rate * gra(X, y, theta)
        cost_history[i] = cost_fonction(X, y, theta)
   
    return theta ,cost_history
    
# Entrainement du model

theta_final,cost_history = Descent_grad(X, y, theta, learning_rate=0.01 , n_iteration=1000)
theta_final

predictions = model(X, theta_final)
plt.scatter(x, y)
plt.plot(x,predictions, c='g')

plt.plot(range(1000),cost_history)


#Evaluation R carre

def coef_determination (y,pred):
    
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    
    return 1 - u/v

print (coef_determination(y, predictions))





s






