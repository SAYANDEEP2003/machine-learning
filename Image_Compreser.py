
# COMPRESSES THE IMAGE BY NEARLY 6 TIMES.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def closest_centroids(X, centroids):
    K = centroids.shape[0]
    index = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
          distance = [] 
          for j in range(centroids.shape[0]):
              distance.append(np.linalg.norm(X[i] - centroids[j]))
          index[i] = np.argmin(distance)
    return index
def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):   
          points = X[idx == k]  
          centroids[k] = np.mean(points, axis = 0)
    return centroids
def KMeans(X, initial_centroids, max_iters=10):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    for i in range(max_iters):
        idx = closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx
def define_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids


original_img=plt.imread("<PATH OF THE FILE(IMAGE)>")
a=original_img
a=a/255
a=np.reshape(a, (a.shape[0] * a.shape[1], 3))
K = 16                       
max_iters = 10               
initial_centroids = define_centroids(a, K) 
centroids, idx = KMeans(a, initial_centroids, max_iters) 
X_recovered = centroids[idx, :]
X_recover = np.reshape(X_recovered, original_img.shape) 
X_recover*255
plt.imshow(X_recover)

