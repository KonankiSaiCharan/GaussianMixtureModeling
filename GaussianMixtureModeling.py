
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np
import random as rd


# In[12]:


def Initialization(GMMData, numOfClusters):
    #Creating centroids with zero values i.e., initializing
    numOfCols = (GMMData.shape)[1]
    covariance = list()
    centroids = np.zeros((numOfClusters, numOfCols))
    pi = np.ones(numOfClusters)*1.0/numOfClusters
    #From the length of the dataset selecting random centroids 
    for numOfClusters in range(numOfClusters):
        index = rd.randint(0,len(GMMData)-1)
        centroids[numOfClusters] = GMMData[index]
        covariance.append(np.cov(GMMData.T))
    return centroids, covariance, pi


# In[23]:


#Here, the sum of all points belonging to this cluster probabilities are calculated
def sumOfResponsibilitiesForCluster(GMMDataX, numOfClusters, centroids, covariance, pi):
    totalProbabilityForCluster = 0.0
    for i in range(numOfClusters):
        totalProbabilityForCluster += pi[i]*GMM(GMMDataX, centroids[i], covariance[i])
    return totalProbabilityForCluster

#Here, we calculate the individual probabilities of each point for a different cluster
def ExpectationStep(GMMData, numOfClusters, centroids, covariance, pi):  
    probabilities = np.zeros((len(GMMData), numOfClusters))
    for index in range(len(GMMData)):
        for j in range(numOfClusters):
            probabilities[index][j] = pi[j] * GMM(GMMData[index], centroids[j], 
                                                       covariance[j])/sumOfResponsibilitiesForCluster(GMMData[index], numOfClusters,
                                                                                                                  centroids, covariance, pi)
    return probabilities

def l_l(GMMData, numOfClusters, centroids, covariance, pi):
    l_l = 0.0
    for x in range (len(GMMData)):
        l_l += np.log(sumOfResponsibilitiesForCluster(GMMData[x], numOfClusters, centroids, covariance, pi))
    return l_l 


# In[25]:


def MaximizationStep(GMMData, numOfClusters, probabilities):
    
    #Getting the dimensions of the data
    NumOfCols = GMMData.shape[1]
    numOfRows = GMMData.shape[0]
    
    #Initializing the centroids to 0
    centroids = np.zeros((numOfClusters, NumOfCols))
    covariance = np.zeros((numOfClusters, NumOfCols, NumOfCols))
    pi = np.zeros(numOfClusters)
    
    #Total probability for eacch cluster is stored in the below variable
    TotalProbForCluster = np.zeros(numOfClusters)
    #By using the prob. values we compute mean, variance and pi using the repective formulae
    for j in range(numOfClusters):
        for i in range(numOfRows):
            TotalProbForCluster[j] += probabilities[i][j]
            centroids[j] += (probabilities[i][j])*GMMData[i]
        centroids[j] = centroids[j]/TotalProbForCluster[j]

        for index in range(numOfRows):
            diffDataCentroids = np.zeros((1,NumOfCols))+GMMData[index]-centroids[j]
            covariance[j] += (probabilities[index][j]/TotalProbForCluster[j])*diffDataCentroids*diffDataCentroids.T
        pi[j] = TotalProbForCluster[j]/numOfRows  
        
    return centroids, covariance, pi


# In[14]:


#Here , we obtain the prob. distribution function of gaussian
def GMM(row, centroids, covariance):
    size = len(row)
    Normalizevalue = (2*np.pi)**size
    Normalizevalue *= np.linalg.det(covariance)
    Normalizevalue = 1.0/np.sqrt(Normalizevalue)
    row_centroids = np.matrix(row-centroids)
    x = Normalizevalue*np.exp(-0.5*row_centroids*np.linalg.inv(covariance)*row_centroids.T)
    return x


# In[37]:


#This is the whole EM loop where we run the E step and the M step repeatedly
def LoopForExpMax(GMMData, numOfClusters, cutoff, Iterations):
    #Take the initial parameters
    IterationLimit = Iterations
    #Initialize the initial parameters from the above functions
    centroids, covariance, pi = Initialization(GMMData, numOfClusters)
    #Calculate the initial log likelihood
    InitialL_L = l_l(GMMData, numOfClusters, centroids, covariance, pi)
    
    #Run through the loop until you get the cutoff value
    for i in range(IterationLimit):
        #Get the probabilities or responsibilities from the expectation step
        probabilities = ExpectationStep(GMMData, numOfClusters, centroids, covariance, pi)
        #Get the updated centroids and covariances from the maximization step
        centroids, covariance, pi = MaximizationStep(GMMData, numOfClusters, probabilities)
        #Updated log likelihood
        updatedL_L = l_l(GMMData, numOfClusters, centroids, covariance, pi)
        #Compare the change with the threshold 
        if (abs(updatedL_L-InitialL_L) < cutoff):
            break
        InitialL_L = updatedL_L
    counts = np.sum(probabilities,axis=0)
    return centroids, covariance, counts


# In[38]:


Data2Gaussiann=np.loadtxt("C:\\Users\\saich\\Desktop\\UnsupervisedML\\2gaussian.txt")
Data3Gaussiann=np.loadtxt("C:\\Users\\saich\\Desktop\\UnsupervisedML\\3gaussian.txt")


# In[39]:


data2Result = LoopForExpMax(GMMData=Data2Gaussiann,Iterations=1000,cutoff=0.1,numOfClusters=2)


# In[40]:


data2Result


# In[41]:


data3Result = LoopForExpMax(GMMData=Data3Gaussiann,Iterations=1000,cutoff=0.1,numOfClusters=3)


# In[42]:


data3Result


# In[46]:


import numpy as np
from sklearn.mixture import GaussianMixture


# In[47]:


#Obtaining the fashion data to perform analysis
def getFashionData(path, kind='train'):
    import numpy as np
    import gzip
    import os
    
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels


# In[48]:


fashionDataValues, fashionDataLabels = getFashionData("C:\\Users\\saich\\Desktop\\UnsupervisedML\\fashion-mnist-master\\data\\fashion", kind='train')


# In[49]:


#Give the num of components and covariance matrix type
GaussianMixtures = GaussianMixture(n_components = 10, covariance_type = 'diag')


# In[50]:


#Fit the model created
GaussianMixtures.fit(fashionDataValues)


# In[51]:


GaussianMixtures.means_


# In[52]:


GaussianMixtures.covariances_


# In[53]:


GaussianMixtures.weights_


# In[54]:


predictedLabels = GaussianMixtures.predict(fashionDataValues[:1000])


# In[67]:


GIImp = list()
maxCluster = list()
clusterDensity = list()
fashionDataLabels2 = fashionDataLabels[:1000]
for IndexOfCluster in range(10):
        Flag = (predictedLabels == IndexOfCluster)
        clusterDensity.append(sum(np.bincount(fashionDataLabels2[Flag])))
        GI = 0
        for i in range(len((np.bincount(fashionDataLabels2[Flag])))):
            GI += (((np.bincount(fashionDataLabels2[Flag]))[i])/sum(np.bincount(fashionDataLabels2[Flag]))) ** 2
        GIImp.append(1 - GI)
        maximum = np.argmax(np.bincount(fashionDataLabels2[Flag]))
        maxCluster.append(np.bincount(fashionDataLabels2[Flag]).max())
maxCluster_sum = sum(maxCluster)


# In[68]:


purityValue = maxCluster_sum/len(fashionDataLabels2)


# In[69]:


purityValue


# In[71]:


GiniUnits = 0
sumValue = 0
for i in range(10):
    sumValue += GIImp[i]*clusterDensity[i]
GiniValue = sumValue/sum(clusterDensity)


# In[72]:


GiniValue


# In[ ]:


#Hence, the purity and gini index obtained using Gaussian mixtures is 0.5 and 0.63.

