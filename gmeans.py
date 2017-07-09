
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from scipy.stats import anderson
import numpy as np
from sklearn.decomposition import PCA

class GMeans(object):
    
    def __init__(self, max_clusters=20, critical_value = 1.159, verbose=0, min_size_ratio=0.05, trim_points=True):
        self.max_clusters = max_clusters
        self.critical_value = critical_value
        self.verbose = verbose
        self.cluster_centers_ = []
        self.labels_ = []
        self.min_size_ratio=min_size_ratio
        self.min_n = 0
        self.trim_points = trim_points
        self.fix_seed = ~trim_points
        self.random_state = 42
        

    def anderson_test(self, X, center):
        pca = PCA(n_components=1)
        pca_data = pca.fit(X)
        m = pca_data.components_ * np.sqrt(2*pca_data.explained_variance_/np.pi)
        init_c = np.zeros((2,2))
        init_c[0,:] = center+m
        init_c[1,:] = center-m

        kmeans = KMeans(n_clusters=2, init = init_c, random_state = self.random_state)
        kmeans.fit(X)
        clusters = kmeans.cluster_centers_

        v = clusters[0] - clusters[1]
        xprime = np.dot(X,v)/np.dot(v,v)
        xprime = scale(xprime)

        # perform Anderson Darling test of normality
        A2 = anderson(xprime)
        
        # modify A2 since mean and std are estimated from the data
        n = len(X)
        A2 = A2.statistic*(1 + 4/n + 25/pow(n,2))
        result = {}
        result['A2'] = A2
        result['kmeans'] = kmeans
        return result
    
    
    def gaussian_test(self, X, center):
        # Ho = The data X is sampled from a Gaussian
        # Ha = The data X is not sampled from a Gaussian

        result_anderson = self.anderson_test(X, center)
        A2 = result_anderson['A2']
        kmeans = result_anderson['kmeans']
        center = kmeans.cluster_centers_
        
        subcluster1 = X[(kmeans.labels_ == 0)]
        subcluster2 = X[(kmeans.labels_ == 1)]
        
        result = {}
        if(A2 < self.critical_value): # Accept null hypothesis, data is sampled from a Gaussian
            result['is_gaussian'] = True
            result['centers'] = center
            
        else:      
            # get A2 score of subclusters
            A2_sc1 = self.anderson_test(subcluster1, kmeans.cluster_centers_[0])['A2']
            A2_sc2 = self.anderson_test(subcluster2, kmeans.cluster_centers_[1])['A2']
            
            # get size of subclusters
            n_sc1 = len(subcluster1)
            n_sc2 = len(subcluster2)

            subcluster1_gaussian = (A2_sc1 < A2) and (n_sc1 > self.min_n)
            subcluster2_gaussian = (A2_sc2 < A2) and (n_sc2 > self.min_n)
            
            
            # only split if splitting is beneficial, i.e. results in subclusters that are 'more' gaussian
            if ( subcluster1_gaussian or subcluster2_gaussian):
                # Reject null hypothesis, data is NOT sampled from a Gaussian. Subclusters are better
                result['is_gaussian'] = False
                result['centers'] = center
                
            else:
                # Cluster is not gaussian, but cannot be split into better gaussians so keep as is
                result['is_gaussian'] = True
                result['centers'] = center
                
        return result

    def fit(self, D):
        self.min_n = self.min_size_ratio * len(D)
        
        
        Dorig = D
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(D)
        clusters = kmeans.cluster_centers_

        gaussian_clusters = list()
        temp_clusters = list()
        temp_clusters.append(clusters[0])

        for k in range(1, self.max_clusters):
            if (len(gaussian_clusters) > self.max_clusters):
                break
            temp_clusters = np.asarray(temp_clusters)
            kmeans = KMeans(n_clusters =len(temp_clusters), init= temp_clusters, random_state = self.random_state)
            kmeans.fit(D)
            all_gaussians = True
            temp_clusters = list()
            klabels = kmeans.labels_

            for l in range(0,len(kmeans.cluster_centers_)):
                # get corresponding center
                center = kmeans.cluster_centers_[l]
                # get corresponding data points
                mask = (klabels==l)
                X = D[mask]
                # perform Gaussian test
                result = self.gaussian_test(X, center)
                if(result['is_gaussian']):
                    #print("Gaussian center exists around: ", center)
                    gaussian_clusters.append(center)
                    # once gaussian center is found, remove the corresponding points and continue
                    if(self.trim_points):
                        D = D[~mask]
                        klabels = klabels[~mask]
                else:
                    #print("Gaussian center doesn't exists around: ", center)
                    all_gaussians = False
                    temp_clusters.append(result['centers'][0])
                    temp_clusters.append(result['centers'][1])

            if (self.verbose):
                print()
                print()
                print("-------------------------------------")
                print("iteration: ",k)
                print("Set of gaussian clusters: ", gaussian_clusters)
                print("Set of temporary clusters: ", temp_clusters)
            if (len(temp_clusters)==0):
                gaussian_clusters = np.asarray(gaussian_clusters)
                break
                
        if (len(gaussian_clusters) == 1):
            self.cluster_centers_ = kmeans.cluster_centers_
            self.labels_ = kmeans.labels_
        else:
            gaussian_clusters = np.asarray(gaussian_clusters)
            kmeans = KMeans(n_clusters =len(gaussian_clusters), init= gaussian_clusters)
            kmeans.fit(Dorig)
            self.cluster_centers_ = kmeans.cluster_centers_
            self.labels_ = kmeans.labels_