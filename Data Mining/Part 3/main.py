# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:52:31 2020

@author: Anne
"""
import pandas as pd
import numpy as np
import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN 
from sklearn.metrics.cluster import contingency_matrix
from scipy.stats import entropy



def binning(insulinwithcarb):
    insulinY = insulinwithcarb['BWZ Carb Input (grams)']
    #print(insulinY)

    minY = min(insulinY)        #3
    maxY = max(insulinY)        #129
    rangeY = (maxY - minY)/20   #6.3 
    print("Range of bins = ",rangeY)
    
    insulinwithcarb['Bin']=""       
    
    for index, row in insulinwithcarb.iterrows():
        if(  (row['BWZ Carb Input (grams)']>=minY) & (row['BWZ Carb Input (grams)']<=minY+21)  ):       
            #row['Bin']=0            # 3 to 23
            insulinwithcarb.at[index,'Bin'] = 0

        if(  (row.loc['BWZ Carb Input (grams)']>=minY+22) & (row.loc['BWZ Carb Input (grams)']<=minY+43)  ):        
            insulinwithcarb.at[index,'Bin'] = 1           #24 to 44
    
        if(  (row['BWZ Carb Input (grams)']>=minY+44) & (row['BWZ Carb Input (grams)']<=minY+65)  ):        
            insulinwithcarb.at[index,'Bin'] = 2            #45 to 65
    
        if(  (row['BWZ Carb Input (grams)']>=minY+66) & (row['BWZ Carb Input (grams)']<=minY+87)  ):        
            insulinwithcarb.at[index,'Bin'] = 3            #66 to 86
    
        if(  (row['BWZ Carb Input (grams)']>=minY+88) & (row['BWZ Carb Input (grams)']<=minY+109)  ):        
            insulinwithcarb.at[index,'Bin'] = 4            #87 to 107

        if(  (row['BWZ Carb Input (grams)']>=minY+110) & (row['BWZ Carb Input (grams)']<=maxY)  ):        
            insulinwithcarb.at[index,'Bin'] = 5            #108 to 129

    insulinwithcarb.to_csv("insulinwithcarb.csv")
        
    return insulinwithcarb

    

def extract_meal_data():
    
    cgm = pd.read_csv('CGMData.csv')
    insulin = pd.read_csv('InsulinData.csv')

    cgm["Date_Time"] = pd.to_datetime(
        cgm["Date"].map(str) + '-' + cgm["Time"])
    insulin["Date_Time"] = pd.to_datetime(
        insulin["Date"].map(str) + '-' + insulin["Time"])
    # CGM data inverted to get it in sequence
    cgm1 = cgm[::-1]
    insulin1 = insulin[::-1]  # Insulin data inverted

    insulinwithcarb = insulin1[insulin1['BWZ Carb Input (grams)'].notnull(
    ) & insulin1['BWZ Carb Input (grams)'] != 0]  # non empty fields in insulin data
    insulinwithcarb.set_index("Index", inplace=True)
    
    prevfoodtime = datetime.datetime.utcfromtimestamp(0)  # init with default date time

    insulinwithcarb.reset_index(drop=True)
        
    #--------------------BINNING--------------------------
    
    insulinwithcarb=binning(insulinwithcarb) 

    #-----------------------------------------------------


    # in each row, get the time diff, if it is greater than or less than 2 hours and based on that remove row if needed
    for index, row in insulinwithcarb.iterrows():
        timedelta = (row['Date_Time'] - prevfoodtime)
        diffmin = timedelta.total_seconds()/60
        if(diffmin > 120):
            prevfoodtime = row['Date_Time']
        else:
            insulinwithcarb.drop(index)

    mealdata = []
    bins=[]

    # Iterate in the cgm data and get data out and store it, storing needs to be checked
    for index, row in insulinwithcarb.iterrows():
        mealdata.append(cgm1.loc[(cgm1['Date_Time'] >= row['Date_Time'] - datetime.timedelta(minutes=30)) &
                                      (cgm1['Date_Time'] <= row['Date_Time'] + datetime.timedelta(hours=2))]['Sensor Glucose (mg/dL)'].to_list())
        bins.append(row['Bin'])
        
    mealDataDF = pd.DataFrame(mealdata)
    binsDF = pd.DataFrame(bins)

    cols = [24, 25, 26, 27, 28, 29, 30]
    mealDataDF.drop(mealDataDF.columns[cols], axis=1, inplace=True)

    indexes_to_drop_meal = []
    for index, row in mealDataDF.iterrows():
        if row.isnull().sum() > 0:
            indexes_to_drop_meal.append(index)

    mealDataDF.drop(indexes_to_drop_meal, axis=0, inplace=True)
    binsDF.drop(indexes_to_drop_meal, axis=0, inplace=True)

    mealDataDF.to_csv("./meal.csv", sep=',', index=False, header=False)
    binsDF.to_csv("./bins.csv", sep=',', index=False, header=False)


    return mealdata,binsDF
    

def normalize(df):
   df=(df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
   return df

    
def features():
        
    meal = pd.read_csv('meal.csv', header=None)
    b = pd.read_csv('bins.csv',header=None)
    
    features = pd.DataFrame()
    fmean=meal.mean(axis=1)
    fmax=meal.max(axis=1)
    fmin=meal.min(axis=1)
    fstd=meal.std(axis=1)
    fvar=meal.var(axis=1)
    fmedian=meal.median(axis=1)
    fskew=meal.skew(axis=1)
    fkurt=meal.kurtosis(axis=1)
    
    features=pd.concat([fmean, fmax,fmin,fstd,fvar,fmedian,fskew,fkurt], axis = 1)
    
    f11=[]
    
    #RMS
    for index,row in meal.iterrows():
        rms = 0
        rms = rms + np.square(index)
        res= np.sqrt(rms / row.shape)
        f11.append(res)
        
    f1 = pd.DataFrame(f11)
    features=pd.concat([features,f1],axis=1)
    features.columns=['mean', 'max','min','std','var','median','skewness','kurtosis','rms']
    
    mealfeatures=normalize(features)
    mealfeatures.insert(len(mealfeatures.columns),'GTruth', b ,True)
        
    km=kmeans(mealfeatures)
    labelsK = km.labels_
    
    mealfeatures.insert(len(mealfeatures.columns),'ClustersK', labelsK ,True)
    mealfeatures.to_csv("features.csv")

    #----------------------------VISUALIZE----------------------------
    #----------------------------KMEANS----------------------------
    print("------------------------------ Before Clustering --------------------------------")

    X = mealfeatures.iloc[:,0] 
    Y = mealfeatures.iloc[:,8] 
    print("\n(Just for visualization (using features 0 and 8))")
    #print("\nBefore Clustering: ")
    plt.ioff()
    
    plt.scatter(X, Y) 
    plt.title('No clusters')
    plt.show()
    
    print("---------------------------------- KMEANS -----------------------------------")
    print("\nLabels for Kmeans = \n", labelsK)

    #print("\nAfter KMeans: ")
    colours = {} 
    colours[0] = 'blue'
    colours[1] = 'red'
    colours[2] = 'pink'
    colours[3] = 'yellow'
    colours[4] = 'green'
    colours[5] = 'purple'
    k=0
    for i in labelsK:
        if i==0:
            plt.scatter(X[k], Y[k], c=colours[0])
        elif i==1:
            plt.scatter(X[k], Y[k], c=colours[1])
        elif i==2:
            plt.scatter(X[k], Y[k], c=colours[2])
        elif i==3:
            plt.scatter(X[k], Y[k], c=colours[3])
        elif i==4:
            plt.scatter(X[k], Y[k], c=colours[4])
        elif i==5:
            plt.scatter(X[k], Y[k], c=colours[5])
        k=k+1
        
    plt.title('KMeans Clusters')
    plt.show()
    #----------------------------- DBSCAN ----------------------------------
    print("---------------------------------- DBSCAN ------------------------------------")
    #print("\nAfter DBSCAN: ")

    db=dbscan(mealfeatures)
    labelsD = db.labels_
    #print("Labels for Dbscan = \n",labelsD)
    
    mealfeatures.insert(len(mealfeatures.columns),'ClustersD', labelsD ,True)
    mealfeatures.to_csv("features.csv")

    """
    colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
    vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
    plt.scatter(X, Y, c=vectorizer(labelsD))
    plt.title('DBSCAN clusters')
    plt.show()
    """
    return km,db, mealfeatures
    #--------------------------------------------------------------------------
    
    
def dbscan(X):
    # Numpy array of all the cluster labels assigned to each data point 
    
    #FINDING EPSILON USING NEAREST NEIGHBOUR
    
    from sklearn.neighbors import NearestNeighbors
    plt.ioff()
    
    neigh = NearestNeighbors(n_neighbors=7)  
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.title('Nearest Neighbour to find Optimal Epsilon (Elbow Method)')
    plt.show()
    plt.close() 

    db = DBSCAN(eps = 0.227, min_samples = 7).fit(X)  #6 clusters!!! 0 to 5
    #0.39 - 22
    #0.3 - 21
    #0.2 - 18
    #db = DBSCAN(eps = 0.148, min_samples = 4).fit(X)  #6 clusters!!! 0 to 5
    #db = DBSCAN(eps = 0.144, min_samples = 4).fit(X)  #6 clusters!!! 0 to 5



    """
    labelsD = db.labels_ 

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(X[['mean', 'max','min','std','var','median','skewness','kurtosis','rms']])
    pca_2d = pca.transform(X[['mean', 'max','min','std','var','median','skewness','kurtosis','rms']])
    for i in range(0, pca_2d.shape[0]):
        if labelsD[i] == 0:
            c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
        elif dbscan.labels_[i] == 1:
            c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
        elif dbscan.labels_[i] == -1:
            c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
    plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
    plt.title('DBSCAN finds 2 clusters and noise')
    plt.show()
    """
    
    return db
    

    
def kmeans(X):
    km = KMeans(n_clusters=6)
    km.fit(X[['mean', 'max','min','std','var','median','skewness','kurtosis','rms']])
    return km


def sseDbscan(labelsD,mealfeatures):
    sse = 0.0
    for idx in range(max(labelsD) + 1):
        rows = [i for i in range(len(labelsD)) if ((labelsD[i] == idx) )]
        mealfeatures = np.array(mealfeatures)
        samples = mealfeatures[rows, :]
        centroid = np.mean(samples, axis = 0)
        for s in samples:
            dist = np.linalg.norm(s - centroid)
            sse += dist * dist
    return sse
    
    
def entropypurity(mealfeatures):
    
    print("----------------------------- Entropy Calculation ---------------------------------\n")
    
    #ENTROPY of KMEANS
    contingencyK=pd.crosstab(mealfeatures['GTruth'],mealfeatures['ClustersK'])
    print(contingencyK)
    contingencyK=np.array(contingencyK)
    entK=entropy(contingencyK,base=2) 
    eK=sum(entK)/6
    #print("Mean entropy (KMeans)= ",eK,"\n")
    
    #PURITY of KMEANS
    numerator=np.sum(np.amax(contingencyK, axis=0)) 
    denominator=np.sum(contingencyK)
    pK=numerator/denominator
    
    #ENTROPY of DBSCAN
    contingencyD=pd.crosstab(mealfeatures['GTruth'],mealfeatures['ClustersD'])
    #contingencyD.drop([-1], axis=1, inplace=True)


    #print(contingencyD)
    contingencyD=np.array(contingencyD)
    entD=entropy(contingencyD,base=2) 
    #print(entD)
    eD=sum(entD)/6
   # print("Mean entropy (Dbscan)= ",eD,"\n")
    
    #PURITY of DBSCAN  
    n=np.sum(np.amax(contingencyD, axis=0)) 
    d=np.sum(contingencyD)
    pD=n/d
    
    return eK, pK, eD, pD
    



if __name__== "__main__":

    mealdata,binsDF=extract_meal_data() 
    km,db,mealfeatures=features()
    
    #--------------Unsupervised Cluster Validity-------------------
    
    sseK = km.inertia_
    sseD = sseDbscan(db.labels_,mealfeatures)
    
    #--------------Supervised Cluster Validity----------------------
    
    eK, pK, eD, pD = entropypurity(mealfeatures)
        
    print("\n------------------------------- Final Results ------------------------------\n")
    print("SSE for Kmeans = ", sseK)
    print("SSE for DBSCAN = ", sseD)

    print("Entropy for Kmeans = ", eK)
    print("Entropy for DBSCAN = ", eD)
    
    print("Purity for Kmeans = ", pK)
    print("Purity for DBSCAN = ", pD)


    resultsList = []
    
    resultsList.append(sseK)
    resultsList.append(sseD)
    resultsList.append(eK)
    resultsList.append(eD)
    resultsList.append(pK)
    resultsList.append(pD)
    
   # dict1 = {'SSE for Kmeans': sseK, 'SSE for DBSCAN': sseD, 'Entropy for Kmeans': eK, 'Entropy for DBSCAN': eD,'Purity for K means':pK, 'Purity for DBSCAN':pD}  
    res = pd.DataFrame([resultsList],columns = ["SSE for KMeans","SSE for DBSCAN","Entropy for KMeans","Entropy for DBSCAN","Purity for KMeans","Purity for DBSCAN"]);

    
    res.to_csv('Results.csv',header=True,index=False)
  






    
