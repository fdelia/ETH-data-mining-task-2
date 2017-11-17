
# coding: utf-8

'''
MapReduce = Training
Transform is called only on the test data
 = Predicting


'''


import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import math

def transform(X_test):
    for i in range(X_test.shape[0]):
        break

    return X_test

# transform = transform_kernel
# transform = transform_nothing

# In[26]:
def kernel(x1, x2):
    return np.inner(x1, x2) 


def mapper(key, value):
    data = []
    for val in value:
        data.append(np.array(val.split(), dtype=float ) )
    
    data = np.array(data)
    
    
    # Training Examples: 
    y = data[:, 0]        # class labels
    x = data[:, 1:]       # features
    
    # Try unconstrained diagonal AdaGrad
    eta = 10               # learning rate parameter
    d = x.shape[1]        # feature dimension
    s = np.ones(d)        # flexible learning rate across dimensions
    w = np.zeros(d)       # SVM vector initialization
    n = data.shape[0]     # number of training examples

    alpha = np.zeros(x.shape[0])
    
    for ite in range(1):
        for i in range(x.shape[0]):
            sm = 0
            for j in range(x.shape[0]):
                sm += alpha[j] * y[j] * kernel(x[i], x[j])

            if sm <= 0:
                val = -1
            if sm > 0:
                val = 1
            if val != y[i]:
                alpha[i] += 1

    print(alpha)
        # s += alpha[i] * y[i] * kernel(x)
        # permutation for SGD
        # perm = np.random.permutation(n)

        # for t in perm:
        #     for i in range(d):
        #         if y[t]*np.dot(x[t, :], w ) < 1 : # also tried as condition: y[t]*x[t, i] < 1
        #             s[i] += (y[t]*x[t, i])**2
        #             w[i] += (eta / np.sqrt(s[i]) ) * y[t] * np.dot(x[t, i], 1) # w/ or w/o '-'
        #             # w[i] += (eta / np.sqrt(s[i]) ) * (1+y[t]*x[t, i])**2
        #             # w[i] += (eta / np.sqrt(s[i]) ) * math.exp(-1 * np.abs(y[t] - x[t, i]) )
            
    # yield 'smile', w
    yield 'smile', alpha


# In[ ]:


def reducer(key, values):
    # IN :
    # key = None
    # values: list of SVM weight vectors (np.array)
    # OUT: 
    # SVM weight vector (np.array)
    # calculated as the average of the SVM weights from values.

    w = np.sum(values, axis=0)
    w = w / len(values)
    print(w)

    yield w

