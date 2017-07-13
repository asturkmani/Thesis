'''

Data Cleaning
    1) Replace all activities that aren't popular by 'other'    
    2) Group entries with the same time stamp into one vector of all activities    
        
------------------------------------------------------------------------------------------------------    
 To get the list of popular activities (i.e. activities that consume time_percentage % of all time spent) we:       
        1) Get statistics of total time spent on each activity    
        2) Find the point where activities have consumed required time by creating CDF over time spent and bisecting at the time_percentage point   
        3) Split the sorted list of activities (by time spent) at the found index     
        4) Once we have the list of popular activities, we replace all unpopular activities by 'other' then add the activity 'other' to the list of popular acitivities
------------------------------------------------------------------------------------------------------     
To group entries:     
        1) First replace all app names by the vector of all apps where each app has a unique index with the apps time spent inserted into its respective place     
        2) Groupby Date and sum across the activity axis to get one vector with all activities
        
'''

import pandas as pd
import numpy as np
import bisect
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import scale

class Clean_DF(object):
    
    def __init__(self, df):
        del df['Number of People']
        self.cdf = []
        self.idx = 0
        self.total_apps = []
        self.popular_apps = []
        self.app_time = []
        self.dirty_df = df
        self.clean_df = self.dirty_df
        self.activity_vector = []
        self.centered_activity_vector = []
        self.pca_data = []
        
    def get_pca(self, explained_variance=0.9, whiten=True, center=True):
        self.activity_vector = np.asarray(self.clean_df['Activity Vector'].tolist())
        self.centered_activity_vector = scale(data_np, with_std=False)
        self.pca = PCA(n_components=explained_variance, whiten=whiten)
        
        if (center):
            self.pca_data = self.pca.fit_transform(self.centered_activity_vector)
        else:
            self.pca_data = self.pca.fit_transform(self.activity_vector)

    def get_day_time(self):
        # Create Day and Time fields
        self.clean_df['Day'] = self.clean_df['Date'].map(lambda p: p.split('T')[0])
        self.clean_df['Time'] = self.clean_df['Date'].map(lambda p: p.split('T')[1])
        
    def clean_data(self, time_percentage=0.9, standardize=True):
        # Normalize time spent across 300 second block
        # Get popular activities
        df_temp = pd.concat([self.dirty_df['Activity'], self.dirty_df['Time Spent (seconds)']], axis=1, keys =['Activity', 'Time Spent (seconds)'])

        # get statistics for Apps
        popular = self.get_popular(df=df_temp, time_percentage=time_percentage)
        self.popular_apps = popular['Popular']
        self.app_cdf = popular['cdf']
        self.app_idx = popular['idx']

        
        # Remove unpopular activities and add 'other' to list of activities
        self.clean_df['Activity'] = self.clean_df['Activity'].map(lambda p : [p] if p in self.popular_apps else ['other'])
        self.clean_df['Category'] = self.clean_df['Category'].map(lambda p: [p])
        self.clean_df['Time Spent (seconds)'] = self.clean_df['Time Spent (seconds)'].map(lambda p: [p])
        self.clean_df['Productivity'] = self.clean_df['Productivity'].map(lambda p: [p])
        
        # Merge activities with same time stamp
        self.clean_df = self.clean_df.reset_index().groupby("Date").sum(axis=1)
        del self.clean_df['index']
        
        # Add column Activity_Vector with vector-encoding of apps and their normalized time spent
        self.clean_df['Activity Vector'] = self.clean_df.apply(lambda x: self.make_array(x), axis=1)
        self.clean_df['Activity Vector'] = self.clean_df['Activity Vector'].map(lambda p: p[0])
        
        # Add idle time to list of activities
        self.clean_df['Activity'] = self.clean_df.apply(lambda x: self.add_idle(x), axis=1)
        self.clean_df['Activity'] = self.clean_df['Activity'].map(lambda p: p[0])
        
        # Create weighted productivity score
        self.clean_df['Productivity Score'] = self.clean_df.apply(lambda x: self.weighted_prod_score(x), axis=1)
        
    def get_popular(self, df, time_percentage=1):
        time_spent_stats = df.groupby('Activity', sort=False).sum().sort_values(by=['Time Spent (seconds)'], ascending=False)
        cdf = time_spent_stats.values.cumsum()/time_spent_stats.values.sum()*np.ones(len(time_spent_stats))
        idx = bisect.bisect(cdf, time_percentage)
        total_apps = list(time_spent_stats.to_dict()['Time Spent (seconds)'].keys())
        popular_apps = time_spent_stats[:idx].to_dict()['Time Spent (seconds)']
        popular_apps['other'] = 0
        popular_apps['idle'] = 0
        popular_apps = list(popular_apps.keys())
        
        return {'Popular': popular_apps, 'cdf': cdf, 'idx': idx}
    
    def add_idle(self, s):
        activity = s['Activity']
        idx = self.popular_apps.index('idle')
        if (s['Activity Vector'][idx] > 0):
            activity.append('idle')
        return [activity]
    
    def weighted_prod_score(self, s):
        score = np.dot(np.array(s['Time Spent (seconds)'])/sum(s['Time Spent (seconds)']), np.array(s['Productivity'])) 
        return score
        
    def make_array(self, s):
        num = len(self.popular_apps)
        z = np.zeros(num)
        time = s['Time Spent (seconds)']
        for i, val in enumerate(s['Activity']):
            idx = self.popular_apps.index(val)
            z[idx] = time[i]
        idx = self.popular_apps.index('idle')
        z = z/300
        if (sum(z) < 1.00):
            z[idx] = 1-sum(z)
        else: #to account for multiple apps running simultaneously
            z = z/sum(z)
        return [z]
        
    def plot_dist(self):
        #Plot distribution of time across activity sorted by most popular
        ts_np1 = self.app_cdf[:self.app_idx]
        plot = sns.barplot(x = np.arange(self.app_idx), y = ts_np1, color = "blue")
        sns.despine(left=True)
        plot.set_ylabel("Percent of time spent")
        plot.set_xlabel("Activities")
        #Set general plot properties
        sns.set_style("white")
        sns.set_context({"figure.figsize": (24, 10)})
        
        
'''

------------------------------ Download rescuetime data ------------------------------

'''

import urllib

def getdata(key, pv="interval" ,rb="2017-01-01", re="2017-07-10", rk = "activity", rs="minute"):
    pv="interval" 
    rk = "activity" 
    format = "csv" 
    rs="minute" 
    rb="2017-01-01" #yyyy-mm-dd from
    re="2017-07-01" #to
    rtapi_key=key #your (looong) API key generated in the Embed and Data API -> Setup Data API
    file_name="data/rescuetime_data-ac-min" #the name of the file where the data will be downloaded
    # get the file
    urllib.request.urlretrieve("https://www.rescuetime.com/anapi/data/?pv="+pv+"&rk="+rk+"&rs="+rs+"&rb="+rb+"&re="+re+"&format="+format+"&rtapi_key="+rtapi_key+"", file_name+".csv")

