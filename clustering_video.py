# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:21:59 2017

@author: dylanrutter
"""
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

data = np.array([[1,2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])
              
plt.scatter(data[:,0], data[:,1], s=150, linewidth=5)
#All the 0th elemments in the X array and all the firstth elements              
plt.show()

colors = ["g.", "r.", "c.", "b.", "k.", "i."]

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        #tol is how much centroid will move by percent change
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    
    def fit(self, data):
        
        self.centroids = {}
        
        for i in range(self.k):
            self.centroids[i] = data[i]
            
        for i in range(self.max_iter):
            self.classifications = {}
            #keys will be centroids and values will be the contained feature sets
     
            for i in range(self.k):
                self.classifications[i] = []
        
            for featureset in data:
                #X is your data
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for\
                centroid in self.centroids]
                #creating a list that is populated with k number of values and 0th index
                #in this list will be the distance to the 0th centroid and 1st ith elemenet
                #will be the distance from that datapoint to centroid 1
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
                prev_centroids = dict(self.centroids)
        
                for classification in self.classifications:
                    self.centroids[classification] = np.average(self.classifications\
                    [classification], axis=0)
            #finds mean of all features and then remakes/redefines centroid
            
            optimized = True
        
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid\
                *100.0) > self.tol:
                    optimized = False
                
            if optimized:
                break
            
            
    
    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for\
        centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = K_Means()
clf.fit(data)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidth=5)
                
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color,
                    s=150, linewidth=5)
                    
plt.show()


