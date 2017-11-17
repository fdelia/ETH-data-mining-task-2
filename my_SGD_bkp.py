
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

# Normalize, no effect
def transform_norm(X):
    for j in range(len(X[0])):
        X[:, j] = X[:, j] - np.mean(X[:, j])

    return X

# Multiple kernels, try out, not optimized
def transform_kernel(X):
    kernel = 'polynomial'
    d = 2
    gamma = 1
    r = 1

    m = 2
    X_new_T = []
    for i in range(m, len(X[0])):
        if kernel == 'polynomial':
            X_new_T.append(X[:, i])
            X_new_T.append(np.power(X[:, i], 2))
            X_new_T.append(np.power(X[:, i], 3))
            # min_val = np.min(X[:, i]); X_new_T.append(np.sqrt(X[:, i] - min_val) )

        for j in range(1, 1 + m):
            # if kernel == 'polynomial':
            #     X_new_T.append( X[:, i] * X[:, i-j])

            if kernel == 'gaussian':
                X_new_T.append( np.exp(-gamma * np.abs(X[:, i] - X[:, i-j])) )        

            if kernel == 'sigmoid':
                X_new_T.append( np.tanh(gamma * X[:, i]* X [:, i-j] + r))
    

    # Remove features with low std
    stds = np.std(X_new_T, axis=1) # Stds of features
    threshold = np.median(stds)
    X_new_T = np.delete(X_new_T, np.argwhere(stds < threshold), axis=0); 

    return np.array(X_new_T).T


# Gaussian kernel, standardized
def transform_kernel_std(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.

    dists = pdist(X, 'sqeuclidean') # pairwise (for every pair of point) euclidiean dist 
    sq_dists = squareform(dists) # make a symmetric matrix
    K = exp(-1 * sq_dists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    eigvals, eigvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    n_components = X.shape[1]
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return X_pc


def transform(X_test):
    for i in range(X_test.shape[0]):
        break

    return X_test

# transform = transform_kernel
# transform = transform_nothing

# In[26]:
def kernel(x1, x2):
    return np.dot(x1, x2) 


def mapper(key, value):
    data = []
    for val in value:
        data.append(np.array(val.split(), dtype=float ) )
    
    data = np.array(data)
    
    # number of training examples
    n = data.shape[0]
    
    # Training Examples: 
    y = data[:, 0]        # class labels
    x = data[:, 1:]       # features
    x = transform(x)
    print(x.shape)
    d = x.shape[1]        # feature dimension
    
    # Try unconstrained diagonal AdaGrad
    eta = 10               # learning rate parameter
    s = np.ones(d)        # flexible learning rate across dimensions
    w = np.zeros(d)       # SVM vector initialization
    alpha = np.zeros(x.shape[0])
    
    for ite in range(1):
        for i in range(x.shape[0]):
            sm = 0
            for j in range(x.shape[0]):
                sm += alpha[j] * y[j] * kernel(x[j], y[j])

            if sm <= 0:
                val = -1
            if sm > 0:
                val = 1
            if val != y[i]:
                alpha[i] += 1

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
            
    yield 'smile', w


# In[ ]:


def reducer(key, values):
    # IN :
    # key = None
    # values: list of SVM weight vectors (np.array)
    # OUT: 
    # SVM weight vector (np.array)
    # calculated as the average of the SVM weights from values.

    s = np.sum(values, axis=0)
    w = s / 1 #len(values)

    yield w

