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
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import scale
import pickle

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
        self.centered_activity_vector = scale(self.activity_vector, with_std=False)
        self.pca = PCA(n_components=explained_variance, whiten=whiten)
        
        if (center):
            self.pca_data = self.pca.fit_transform(self.centered_activity_vector)
        else:
            self.pca_data = self.pca.fit_transform(self.activity_vector)

    def get_day_time(self):
        # Create Day and Time fields
        self.clean_df['Day'] = self.clean_df['Date'].map(lambda p: p.split('T')[0])
        self.clean_df['Time'] = self.clean_df['Date'].map(lambda p: p.split('T')[1])
        
    def clean_data(self, time_percentage=0.9, standardize=True, data_cat='Category', add_idle=False):
        # Normalize time spent across 300 second block
        # Get popular activities
        
        df_temp = pd.concat([self.dirty_df[data_cat], self.dirty_df['Time Spent (seconds)']], axis=1, keys =[data_cat, 'Time Spent (seconds)'])
        # get statistics for Apps
        popular = self.get_popular(df=df_temp, time_percentage=time_percentage)
        self.popular_apps = popular['Popular']
        self.app_cdf = popular['cdf']
        self.app_idx = popular['idx']

        
        # Remove unpopular activities and add 'other' to list of activities
        
        self.clean_df[data_cat] = self.clean_df[data_cat].map(lambda p : [p] if p in self.popular_apps else ['other'])
        
        if (data_cat == 'Activity'):
            self.clean_df['Category'] = self.clean_df['Category'].map(lambda p: [p])
            self.clean_df['Productivity'] = self.clean_df['Productivity'].map(lambda p: [p])
            
        self.clean_df['Time Spent (seconds)'] = self.clean_df['Time Spent (seconds)'].map(lambda p: [p])
        
        
        # Merge activities with same time stamp
        self.clean_df = self.clean_df.reset_index().groupby("Date").sum(axis=1)
        del self.clean_df['index']
        
        # Add column Activity_Vector with vector-encoding of apps and their normalized time spent
        self.clean_df['Activity Vector'] = self.clean_df.apply(lambda x: self.make_array(x), axis=1)
        self.clean_df['Activity Vector'] = self.clean_df['Activity Vector'].map(lambda p: p[0])
        
        # Add idle time to list of activities
        if (add_idle):
            self.clean_df[data_cat] = self.clean_df.apply(lambda x: self.add_idle(x, data_cat), axis=1)
            self.clean_df[data_cat] = self.clean_df[data_cat].map(lambda p: p[0])
        
        self.activity_vector = np.asarray(self.clean_df['Activity Vector'].tolist())
        # Create weighted productivity score
        if (data_cat == 'Activity'):
            self.clean_df['Productivity Score'] = self.clean_df.apply(lambda x: self.weighted_prod_score(x), axis=1)
        
    def get_popular(self, df, time_percentage=1, data_cat='Category', add_idle=False):
        time_spent_stats = df.groupby(data_cat, sort=False).sum().sort_values(by=['Time Spent (seconds)'], ascending=False)
        cdf = time_spent_stats.values.cumsum()/time_spent_stats.values.sum()*np.ones(len(time_spent_stats))
        idx = bisect.bisect(cdf, time_percentage)
        total_apps = list(time_spent_stats.to_dict()['Time Spent (seconds)'].keys())
        popular_apps = time_spent_stats[:idx].to_dict()['Time Spent (seconds)']
        popular_apps['other'] = 0
        
        if(add_idle):
            popular_apps['idle'] = 0
        popular_apps = list(popular_apps.keys())
        
        return {'Popular': popular_apps, 'cdf': cdf, 'idx': idx}
    
    def add_idle(self, s, data_cat = 'Category'):
        activity = s[data_cat]
        idx = self.popular_apps.index('idle')
        if (s['Activity Vector'][idx] > 0):
            activity.append('idle')
        return [activity]
    
    def weighted_prod_score(self, s):
        score = np.dot(np.array(s['Time Spent (seconds)'])/sum(s['Time Spent (seconds)']), np.array(s['Productivity'])) 
        return score
        
    def make_array(self, s, data_cat = 'Category', add_idle=False):
        num = len(self.popular_apps)
        z = np.zeros(num)
        time = s['Time Spent (seconds)']
        for i, val in enumerate(s[data_cat]):
            idx = self.popular_apps.index(val)
            z[idx] = time[i]
        
        if (add_idle):
            idx = self.popular_apps.index('idle')
        z = z/300
        if (sum(z) < 1.00):
            if (add_idle):
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

def getdata(key, pv="interval" ,rb="2017-01-01", re="2017-07-18", rk = "activity", rs="minute"):
    format = "csv" 
    rtapi_key=key #your (looong) API key generated in the Embed and Data API -> Setup Data API
    file_name="data/rescuetime_data_" + rk + "_"+ re #the name of the file where the data will be downloaded
    # get the file
    urllib.request.urlretrieve("https://www.rescuetime.com/anapi/data/?pv="+pv+"&rk="+rk+"&rs="+rs+"&rb="+rb+"&re="+re+"&format="+format+"&rtapi_key="+rtapi_key+"", file_name+".csv")

'''

---- Data Preprocessing ---

'''

def split2sequences(data, length_x=1, length_y=1, split=0.8):
    print('Splitting text into sequences...', "\n",)
    step = 1
    xN = []
    yN = []
    
    for i in range(0, len(data) - length_x, step):
        xN.append(data[i: i + length_x])
        yN.append(data[i+1+length_x-length_y:i + length_x + 1])
        
    train_size = int(len(xN) * split)
    test_size = len(xN) - train_size
    
    xN = np.array(xN)
    yN = np.array(yN)
    n = len(data)
    X_train, X_test = xN[0:train_size],  xN[train_size:n]
    Y_train,Y_test = yN[0:train_size], yN[train_size:n]

    return xN, yN, X_train, X_test, Y_train, Y_test

def make_timeseries_instances(timeseries, window_size, single_y=False):
    """Make input features and prediction targets from a `timeseries` for use in machine learning.
    :return: A tuple of `(X, y, q)`.  `X` are the inputs to a predictor, a 3D ndarray with shape
      ``(timeseries.shape[0] - window_size, window_size, timeseries.shape[1] or 1)``.  For each row of `X`, the
      corresponding row of `y` is the next value in the timeseries.  The `q` or query is the last instance, what you would use
      to predict a hypothetical next (unprovided) value in the `timeseries`.
    :param ndarray timeseries: Either a simple vector, or a matrix of shape ``(timestep, series_num)``, i.e., time is axis 0 (the
      row) and the series is axis 1 (the column).
    :param int window_size: The number of samples to use as input prediction features (also called the lag or lookback).
    """
    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]
    X = np.atleast_3d(np.array([timeseries[start:start + window_size] for start in range(0, timeseries.shape[0] - window_size)]))
    y = timeseries[window_size:]
    q = np.atleast_3d([timeseries[-window_size:]])
    return X, y, q

def convert_to_rgb(x):
    if (x.ndim > 1):
        c=['rgb'+str(tuple(i)) for i in x.tolist()]
    else:
        c = 'rgb'+str(tuple(x))
    return c

def make_clean_data(window_size,batch_size=0, val_size=0.2,multiplier=300, process = False, time_percentage=0.9, explained_variance=0.9):
    filename = 'data.pickle'
    delete_other_idle = False
    if (process):
        df = pd.read_csv('rescuetime_data_category_2017-07-21.csv')
        data = Clean_DF(df)
        data.clean_data(time_percentage=time_percentage)
        data.clean_df = data.clean_df.reset_index()
        data.get_pca(explained_variance=explained_variance)
        data.get_day_time()

        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(data, f)
    else:
        with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
            data = pickle.load(f)

    dataset = data.activity_vector
    popular_apps = data.popular_apps


    if (delete_other_idle):
    # Remove IDLE and OTHER time
        dataset = np.delete(dataset, [dataset.shape[1]-1,dataset.shape[1]-2], axis=1)
        del popular_apps[-1]
        del popular_apps[-1]



    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday','Saturday','Sunday']


    days = set(data.clean_df['Day'])
    df = data.clean_df[['Date', 'Activity Vector']]
    df['timestamp'] = pd.to_datetime(df['Date'])
    # print(df.dtypes)
    df = df.set_index('timestamp').resample('300S').asfreq()
    x = {'val': np.zeros(15)}
    df['Activity Vector'] = df['Activity Vector'].fillna(x)
    del df['Date']
    df['Activity Vector'] = df['Activity Vector'].map(lambda x: np.zeros(len(popular_apps)) if np.isnan(np.sum(x)) else x)
    df = df.reset_index()
    dataset = df['Activity Vector']
    dataset = np.asarray(dataset.tolist())

    day_categorical = np.asarray(df.timestamp.dt.weekday)
    time_categorical = np.asarray(df.timestamp.dt.time.map(lambda x: int(str(x)[0:2])))
    minute_categorical = np.asarray(df.timestamp.dt.time.map(lambda x: int(str(x)[3:5])/5))

    (X,y,q) = make_timeseries_instances(timeseries=dataset*multiplier, window_size=window_size)
    (Xd,yd,qd) = make_timeseries_instances(timeseries=day_categorical, window_size=window_size)
    (Xt,yt,qt) = make_timeseries_instances(timeseries=time_categorical, window_size=window_size)
    (Xm,ym,qm) = make_timeseries_instances(timeseries=minute_categorical, window_size=window_size)
    
    assert(len(y) == len(yd) == len(yt))
    print(Xd.shape, yd.shape, Xt.shape, yt.shape, Xm.shape, ym.shape)
    indices = ~np.all(y == 0, axis=1)
    
    Xc = X[indices, :, :]
    yc = y[indices, :]
    
    Xd = Xd[indices,:]
    yd = yd[indices]
    
    Xt = Xt[indices,:]
    yt = yt[indices]
    
    Xm = Xm[indices,:]
    ym = ym[indices]
    
    test_size = int(val_size * Xc.shape[0])           # In real life you'd want to use 0.2 - 0.5
    x_train_c, x_test_c, y_train_c, y_test_c = Xc[:-test_size], Xc[-test_size:], yc[:-test_size], yc[-test_size:]
    Xt = Xt.reshape(Xt.shape[0],Xt.shape[1])
    Xd = Xd.reshape(Xt.shape[0],Xt.shape[1])
    Xm = Xm.reshape(Xm.shape[0],Xm.shape[1])
    
    x_train_t, x_test_t, y_train_t, y_test_t = Xt[:-test_size], Xt[-test_size:], yt[:-test_size], yt[-test_size:]
    x_train_d, x_test_d, y_train_d, y_test_d = Xd[:-test_size], Xd[-test_size:], yd[:-test_size], yd[-test_size:]
    x_train_m, x_test_m, y_train_m, y_test_m = Xm[:-test_size], Xm[-test_size:], ym[:-test_size], ym[-test_size:]
    
    if (batch_size > 0):
        l_total = int(len(Xc)/batch_size)*batch_size
        l_train = int(len(x_train_c)/batch_size)*batch_size
        l_test = int(len(x_test_c)/batch_size)*batch_size

        Xc = Xc[:l_total]
        yc = yc[:l_total]
        Xd = Xd[:l_total]
        yd = yd[:l_total]
        Xt = Xt[:l_total]
        yt = yt[:l_total]
        Xm = Xm[:l_total]
        ym = ym[:l_total]

        x_train_c = x_train_c[:l_train]
        x_test_c = x_test_c[:l_test]
        y_train_c = y_train_c[:l_train]
        y_test_c = y_test_c[:l_test]

        x_train_t = x_train_t[:l_train]
        x_test_t = x_test_t[:l_test]
        y_train_t = y_train_t[:l_train]
        y_test_t = y_test_t[:l_test]

        x_train_d = x_train_d[:l_train]
        x_test_d = x_test_d[:l_test]
        y_train_d = y_train_d[:l_train]
        y_test_d = y_test_d[:l_test]
        
        x_train_m = x_train_m[:l_train]
        x_test_m = x_test_m[:l_test]
        y_train_m = y_train_m[:l_train]
        y_test_m = y_test_m[:l_test]
    
    y_train_labels = [dict(zip(popular_apps, np.round(300*x))) for x in y_train_c]
    y_test_labels = [dict(zip(popular_apps, np.round(300*x))) for x in y_test_c]
    y_labels = [dict(zip(popular_apps, np.round(300*x))) for x in yc]

    cmap = {
        'Instant Message' : (255,255,0),
        'Video' : (255,0,0),
        'General Software Development' : (255,0,200),
        'General Social Networking' : (255,127,0),
        'Writing' : (200,0,255),
        'Browsers' : (0,255,200),
        'General Reference & Learning' : (127,0,255),
        'Email' : (127,255,0),
        'Search' : (0,255,127),
        'Uncategorized' : (128,128,128),
        'General News & Opinion' : (0,200,255),
        'Engineering & Technology' : (0,127,255),
        'General Business' : (0,0,255),
        'Voice Chat' : (200,255,0),
        'other' : (200,200,200)
    }
    app_colors = ['green','red', 'orange', 'magenta','cerulean', 'yellow', 'light purple', 'turquoise', 'tan', 'grey', 'maroon', 'gold', 'dark green', 'navy blue', 'brown']
    col_list_palette = sns.xkcd_palette(app_colors)
    yc_colors = np.einsum('ij,jk->ik', yc, col_list_palette)
    y_train_c_colors = np.einsum('ij,jk->ik', y_train_c, col_list_palette)
    y_test_c_colors = np.einsum('ij,jk->ik', y_test_c, col_list_palette)
    data_colors = col_list_palette
    
    data_ = {
        'Xc' : Xc,
        'yc' : yc,
        'Xt' : Xt,
        'yt' : yt,
        'Xd' : Xd,
        'yd' : yd,
        'Xm' : Xm,
        'ym' : ym,
        'x_train_c' : x_train_c,
        'x_test_c' : x_test_c,
        'y_train_c' : y_train_c,
        'y_test_c' : y_test_c,
        'x_train_t' : x_train_t,
        'x_test_t' : x_test_t,
        'y_train_t' : y_train_t,
        'y_test_t' : y_test_t,
        'x_train_d' : x_train_d,
        'x_test_d' : x_test_d,
        'y_train_d' : y_train_d,
        'y_test_d' : y_test_d,
        'x_train_m' : x_train_m,
        'x_test_m' : x_test_m,
        'y_train_m' : y_train_m,
        'y_test_m' : y_test_m,
        'popular_apps' : popular_apps,
        'days' : len(set(day_categorical)),
        'time' : len(set(time_categorical)),
        'minute' : len(set(minute_categorical)),
        'day_categorical' : day_categorical,
        'time_categorical' : time_categorical,
        'dataset' : dataset,
        'y_train_labels' : y_train_labels,
        'y_test_labels' : y_test_labels,
        'y_labels' : y_labels,
        'yc_colors' : yc_colors,
        'y_train_c_colors' : y_train_c_colors,
        'y_test_c_colors' : y_test_c_colors,
        'data_colors' : data_colors,
        'cmap' : cmap
    }
    
    return data_