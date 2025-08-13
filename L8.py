import numpy as np
data=np.array([5,15,10,20,25,30,25,35])

#Mean
mean=np.mean(data)
print('Mean:',mean)

#Median
median=np.median(data)
print('Median:',median)

#Mode
from scipy import stats
mode=stats.mode(data)
print("Mode:",mode[0])

#Show dictonary
dir(np)
import scipy as sc
dir(sc)

#SD
std_dev=np.std(data)
print("Standard Deviation:",std_dev)

#Variance
variance=np.var(data)
print('Variance',variance)

#Skewnwess ,Kurtosis
from scipy.stats import skew,kurtosis
Skewnwess=stats.skew(data)
Kurtosis=stats.kurtosis(data)
print('Skewnwess:',Skewnwess)
print('Kurtosis:',Kurtosis)

X=np.array([1,2,3,4,5])
Y=np.array([2,4,5,4,5])
correlation_matrix=np.corrcoef(X,Y)
correlation= correlation_matrix[0,1]
print("Pearson correlation coefficient:",correlation)
mean_X= np.mean(X)
mean_Y= np.mean(Y)