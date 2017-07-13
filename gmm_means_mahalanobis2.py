
'''
    1) Run k-means on sub-data X with k=2      
    2) Compute v = c1 - c2 where c1 & c2 are the two new cluster centres found by k-means on X 
    3) Project X onto v Xv = dot(X,v)/dot(v,v)      
    4) Scale Xv to mean 0 and unit variance      
    5) apply Anderson-Darling test to this vector 
'''

'''
    Problems with G-Means:
    1) Re-running k-means gets stuck in loop:
    solution: remove points that have been classified, OR, fix k-means Seed
    
    2) Splitting non-gaussians results in infinite splits if data isn't normally distributed
    solution: Only split if subclusters are more gaussian and above min_size threshold

'''
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale
from scipy.stats import anderson
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

class GMMMeans(object):
    def __init__(self, critical_value = 1.159, verbose=0, center_init='pca', recalculate_points = True, max_iter=50,min_cluster_size=200):
        self.max_iter = max_iter
        self.critical_value = critical_value
        self.verbose = verbose
        self.center_init = center_init
        self.recalculate_points = recalculate_points
        self.min_cluster_size = min_cluster_size
        self.cluster_centers_ = []
        self.labels_ = []
      

    # Find all points in the dataset that are within a Mahalonobis distance of 2 (or 2 stds in the univariate case)
    def possible_points(self, M, V):
        mdist = cdist(self.D, [M], metric='mahalanobis', V=V)[:,0]
        d2_mask = mdist < 2  
        x = self.D[d2_mask,:]
        return x
        
    def pca_sub_centers(self, X, center):
        
        pca = PCA(n_components=1)
        pca_data = pca.fit(X)
        
        m = pca_data.components_ * np.sqrt(2*pca_data.explained_variance_/np.pi)
        
        init_centers = np.zeros((2,m.shape[1]))
        center = np.reshape(center, (1,m.shape[1]))
        
        init_centers[0,:] = center+m
        init_centers[1,:] = center-m
        
        return init_centers
        
    def sub_anderson_test(self, X, center):
        
        if(self.center_init=='pca'):
            init_centers = self.pca_sub_centers(X, center)
            gmm = GaussianMixture(n_components=2, means_init = init_centers)
        else:
            gmm = GaussianMixture(n_components=2)
            
        gmm.fit(X)
        sub_centers = gmm.means_
        
        v = sub_centers[0] - sub_centers[1]
        xprime = np.dot(X,v)/np.dot(v,v)
        xprime = scale(xprime)
        
        A2 = anderson(xprime)
        n = len(X)
        A2 = A2.statistic*(1 + 4/n - 25/pow(n,2))
        
        result = {}
        result['A2'] = A2
        result['sub_centers'] = sub_centers
        result['labels'] = gmm.predict(X)
        return result
    
    def anderson_test(self, X, center):  
    
        if (self.recalculate_points):
            M = np.mean(X, axis=0)
            V = np.cov(X, rowvar=0)
            possibleX = self.possible_points(M,V)
            if (len(possibleX) > self.min_cluster_size):
                result = self.sub_anderson_test(possibleX, center)
            else:
                result = {}
                result['A2'] = 10000
                result['sub_centers'] = []
                result['labels'] = []
                
        else:
            result = self.sub_anderson_test(X, center)
            
        return result
    
    def gaussian_test(self, X, center):
        # Ho = The data X is sampled from a Gaussian
        # Ha = The data X is not sampled from a Gaussian

        result = self.anderson_test(X, center)
        A2 = result['A2']
        sub_centers = result['sub_centers']
        labels_ = result['labels']
      
        result = {}
        if(A2 < self.critical_value): # Accept null hypothesis, data is sampled from a Gaussian
            result['is_gaussian'] = True
            result['centers'] = center
        else:
            result['is_gaussian'] = False
            result['centers'] = sub_centers
        return result
        
    def fit(self, D):
        #self.min_n =len(D)/self.max_clusters
        self.D = D

        gmm = GaussianMixture(n_components=1)
        gmm.fit(D)
        
        first_cluster = gmm.means_
        labels = gmm.predict(self.D)
        
        prev_number_temp_clusters = 0
        temp_clusters = np.zeros((0, D.shape[1]))
        temp_clusters = np.concatenate((temp_clusters,first_cluster))
        self.cluster_centers = np.zeros((0, D.shape[1]))
        
        i=0
        while (len(temp_clusters) > prev_number_temp_clusters and i <self.max_iter):
            i +=1
            prev_number_temp_clusters = len(temp_clusters)
            
            if (self.verbose):
                print()
                print()
                print("-------------------------------------")
                print("Set of clusters to test: ", temp_clusters)

            
            gmm = GaussianMixture(n_components=len(temp_clusters), means_init = temp_clusters)
            gmm.fit(D)
            
            refined_clusters = gmm.means_
            refined_labels = gmm.predict(D)
            
            temp_clusters = np.zeros((0, D.shape[1]))
            
            for i in range(0,len(refined_clusters)):
                center_to_test = refined_clusters[i]
                x_to_test = self.D[refined_labels==i]
                
                if (len(x_to_test) > self.min_cluster_size):
                    result = self.gaussian_test(x_to_test, center_to_test)
                    if (result['is_gaussian'] and len(x_to_test)>5):
                        center_to_test = np.reshape(center_to_test, (1,temp_clusters.shape[1]))
                        self.cluster_centers = np.concatenate((self.cluster_centers,center_to_test))
                    else:
                        if (len(result['centers']) > 0):
                            if ~(np.round(result['centers'][0]) in np.round(self.cluster_centers)):
                                center = np.reshape(result['centers'][0], (1,temp_clusters.shape[1]))
                                temp_clusters = np.concatenate((temp_clusters,center))
                            if ~(np.round(result['centers'][1]) in np.round(self.cluster_centers)):
                                center = np.reshape(result['centers'][1], (1,temp_clusters.shape[1]))
                                temp_clusters = np.concatenate((temp_clusters,center))
                    
    
        gmm = GaussianMixture(n_components=len(temp_clusters), means_init = temp_clusters)
        gmm.fit(D)
        self.cluster_centers_ = gmm.means_
        self.labels_ = gmm.predict(self.D)
            