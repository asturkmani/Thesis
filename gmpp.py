
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
from scipy.stats import f

class GMeans(object):
    """ G-Means+ clustering
    
    Parameters
    ----------
    
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
        
    max_clusters : int, optional, default: 20
        Maximum number of clusters G-Means should find after 
        which it should terminate

    critical_value = float, default: 1.159
        Critical value Anderson Darling for tests. Determined from required level of significance.
        See https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test for more info

    verbose : int, default 0
        Verbosity mode.
                      
    min_size_ratio = float, optional, default: 0.05
        Minimum ratio of entire dataset that can belong to a cluster. 
        No cluster sizes created with less than min_size_ratio * len(X) elements
                             
    trim_points = boolean, optional, default: False
        Remove points from input data that have already been assigned to a cluster.
        (Helps with convergence in data with heavily overlapping distributions)
        
         
    center_init = {'default', 'pca'}
        Determines how K-Means centers are initialised when performing Anderson Darling test
        
        'default' : initialise clusters using K-Means default settings
        
        'pca' : centers initialised as follows
            
            sub_center1 = center + m
            sub_center2 = center - m
            m = s * sqrt(2*lambda/pi)
            s = 1st principal component of passed dataset
            lambda = corresponding eigenvalue
                              
    recalculate_points = boolean, default: True
        Determine set of points over which to conduct Anderson Darling test.
        (Setting to True helps cluster datasets with substantial overlap in distributions
        
        'True' :  Re-calculate points that belong to the Gaussian center being tested.
            This is done by first estimating the clusters standard deviation via the points 
            attributed to that cluster, then projecting entire dataset onto the first 
            principal component of subset. All points within 2 standard deviations of the mean
            are re-assigned to that cluster for the Anderson Darling test
            
         'False' : Points are not re-assigned and passed subset is used instead.
         
    significance_level = float < 1, default: 0.95
        The level of significance at which the Fisher test is conducted to re-assign points to Gaussian centers
        whenever a center is tested via Anderson Darling
        
        Note: only used when recalculate_points = True
            
    Attributes
    ----------
    
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
        
    labels_ :
        Labels of each point
        """
    
    def __init__(self, max_clusters=10, critical_value = 1.859, verbose=0, trim_points=True, center_init='random', recalculate_points = True, significance_level=0.99):
        self.max_clusters = max_clusters
        self.critical_value = critical_value
        self.verbose = verbose
        self.cluster_centers_ = []
        self.labels_ = []

        self.min_n = 0
        self.trim_points = trim_points
        self.fix_seed = ~trim_points
        self.random_state = 42
        self.center_init = center_init
        self.recalculate_points = recalculate_points
        self.significance_level = significance_level
        self.F = 0
        self.k = 0
      

    def fisher_test(self, xmu, mean, inv_cov):
        val = np.dot(np.dot(xmu,inv_cov), np.transpose(xmu))
        return self.k*val < self.F
    
    
    def anderson_test(self, X, center):        
        # Select points that pass the Fisher test at 0.95 level of significance
        fX = X
        if (self.recalculate_points):
            mean = np.mean(X, axis=0)
            cov = np.cov(X, rowvar=0)
            inv_cov = np.linalg.inv(cov)
            #print("length of D: ", len(self.D))
            #print("mean of X: ", mean)
            xmu = self.D - mean
            # apply fisher test to all values in D to get points that could belong to this gaussian
            indices = np.apply_along_axis(self.fisher_test, 1, xmu, mean, inv_cov)
            #print("indices shape: ", indices.sum())
            fX = self.D[indices]
            
        pca = PCA(n_components=1)
        pca_data = pca.fit(fX)
        m = pca_data.components_ * np.sqrt(2*pca_data.explained_variance_/np.pi)
        center = np.reshape(center, (1,m.shape[1]))
        init_c = np.zeros((2,m.shape[1]))
        init_c[0,:] = center+m
        init_c[1,:] = center-m
            
        
        # Initialize k-means as specified
        if(self.center_init=='pca'):
            kmeans = KMeans(n_clusters=2, init = init_c)
        elif(self.center_init=='random'):
            kmeans = KMeans(n_clusters=2)
        else:
            raise ValueError(self.center_init, " not a valid k-means initialisation method")

        kmeans.fit(fX)
        clusters = kmeans.cluster_centers_

        v = clusters[0] - clusters[1]
        #print("v: ", v)
        #print("pca comp1: ", m)
        xprime = np.dot(fX,v)/np.dot(v,v)
        xprime = scale(xprime)

        # perform Anderson Darling test of normality
        A2 = anderson(xprime)
        
        # modify A2 since mean and std are estimated from the data
        
        # Return child clusters from X, and not from D
        pca_data = pca.fit(X)
        m = pca_data.components_ * np.sqrt(2*pca_data.explained_variance_/np.pi)
        center = np.reshape(center, (1,m.shape[1]))
        init_c = np.zeros((2,m.shape[1]))
        init_c[0,:] = center+m
        init_c[1,:] = center-m
        kmeans = KMeans(n_clusters=2, init = init_c)
        kmeans.fit(X)
        
        n = len(fX)
        A2 = A2.statistic*(1 + 4/n + 25/pow(n,2))
        result = {}
        result['A2'] = A2
        result['kmeans'] = kmeans
        result['X'] = X
        return result
    
    
    def gaussian_test(self, X, center):
        # Ho = The data X is sampled from a Gaussian
        # Ha = The data X is not sampled from a Gaussian

        result_anderson = self.anderson_test(X, center)
        A2 = result_anderson['A2']
        kmeans = result_anderson['kmeans']
        X = result_anderson['X']
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
        self.min_n =len(D)/self.max_clusters
        self.D = D
        
        # parameters for recalculation
        
        if (self.recalculate_points):
            n = self.D.shape[0]
            p = self.D.shape[1]
            k = n*(n-p)/( p*(n-1)*(n+1))
            self.k = k
            self.F = f.ppf(self.significance_level, p, n-p , loc=0, scale=1)

        kmeans = KMeans(n_clusters=1)
        kmeans.fit(D)
        clusters = kmeans.cluster_centers_

        gaussian_clusters = np.zeros((0, D.shape[1]))
        temp_clusters = np.zeros((0, D.shape[1]))
        temp_clusters = np.concatenate((temp_clusters,clusters))

        for k in range(1, self.max_clusters):
            
            if (len(gaussian_clusters) > self.max_clusters):
                break
            
            all_centers = np.concatenate((gaussian_clusters,temp_clusters))
            kmeans = KMeans(n_clusters =len(all_centers), init= all_centers)
            kmeans.fit(D)
            all_gaussians = True
            temp_clusters = np.zeros((0, D.shape[1]))
            klabels = kmeans.labels_
            
            if (self.verbose):
                print()
                print()
                print("-------------------------------------")
                print("iteration: ",k)
                print("Set of gaussian clusters: ", gaussian_clusters)
                print("Set of temporary clusters: ", temp_clusters)

            #if (np.round(center) is in np.round(gaussian_clusters))
            for l in range(0,len(kmeans.cluster_centers_)):
                # get corresponding center
                center = kmeans.cluster_centers_[l]
                #print(" center found by k-means: ", center)
                if (np.round(center) in np.round(gaussian_clusters)):
                    #print("center: ", center," already found ")
                    continue
                # get corresponding data points
                mask = (klabels==l)
                X = D[mask]
                # perform Gaussian test
                result = self.gaussian_test(X, center)
                sub_centers = result['centers']
                sub_centers = np.asarray(sub_centers).reshape((len(sub_centers),D.shape[1]))
                
                if(result['is_gaussian']):
                    #print("Gaussian center exists around: ", center)
                    center = np.asarray(center).reshape((1,D.shape[1]))
                    gaussian_clusters = np.concatenate((gaussian_clusters,center))
                    # once gaussian center is found, remove the corresponding points and continue
                    if(self.trim_points):
                        #print("Trimming found cluster")
                        #print(D.shape)
                        D = D[~mask]
                        #print("After trim: ", D.shape)
                        klabels = klabels[~mask]
                else:
                    #print("Gaussian center doesn't exists around: ", center)
                    all_gaussians = False
                    temp_clusters = np.concatenate((temp_clusters,sub_centers))

            if (len(temp_clusters)==0):
                break
                
        if (len(gaussian_clusters) == 1):
            self.cluster_centers_ = kmeans.cluster_centers_
            self.labels_ = kmeans.labels_
        else:
            kmeans = KMeans(n_clusters =len(gaussian_clusters), init= gaussian_clusters)
            kmeans.fit(self.D)
            self.cluster_centers_ = kmeans.cluster_centers_
            self.labels_ = kmeans.labels_