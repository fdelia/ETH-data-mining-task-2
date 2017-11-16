from __future__ import division
import numpy as np
import time as tp
from matplotlib.mlab import PCA


def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # def make_bins(x):
    #     features = []
    #     for s in np.array_split(x, 50):
    #         features.append(np.histogram(s, bins=bins)[0])
    #     return np.array(features).ravel()
    #
    # X = np.apply_along_axis(make_bins, 1, X)



    if np.ndim(X) == 1:
        X = np.reshape(X, (1, len(X) ) )

    n, d = X.shape
    m = 9600
    gamma = 17.625
    np.random.seed(1)
    omega = np.sqrt(2*gamma)*np.random.standard_normal((d, m) )
    b = np.random.random_sample(m) * 2 * np.pi

    Z = np.sqrt(2 / float(m)) * np.cos(np.add(np.dot(X, omega), b ) )
    return Z

# In[26]:


def mapper(key, value):

    data = []
    for val in value:
        data.append(np.array(val.split(), dtype=float ) )

    data = np.array(data)
    # number of training examples
    n = data.shape[0]

    # permutation for SGD
    perm = np.random.permutation(n)

    # Training Examples:
    y = data[:, 0]        # class labels
    x = data[:, 1 : ]     # features
    z = transform(x)
    d = z.shape[1]        # feature dimension



    # Try ADAM
    m = v = w = np.zeros(d)   	# SVM vector initialization
    alpha = 1 		# stepsize parameter
    beta_1 = 0.9
    beta_2 = 0.999
    #beta_1_arr = [0.5,0.85,0.9,0.95]				# momentum
    #beta_2_arr = [0.55,0.9,0.95,0.99]				# momentum
    epsilon = 1e-8

    for p, t in enumerate(perm):
            #if t<500:
            #    beta_1 = beta_1_arr[0]
            #    beta_2 = beta_2_arr[0]
            #if t>=500 and t<1000:
            #    beta_1 = beta_1_arr[1]
            #   beta_2 = beta_2_arr[1]
            #if t>=1000 and t<1500:
            #    beta_1 = beta_1_arr[2]
            #    beta_2 = beta_2_arr[2]
            #if t>=1500 and t<2000:
            #    beta_1 = beta_1_arr[3]
            #    beta_2 = beta_2_arr[3]
            if y[t]*np.dot(z[t, :], w ) < 1 :
            	g = (-1)*y[t] * z[t, :]
            	m = beta_1 * m + (1 - beta_1 ) * g					# momentum term 1
            	v = beta_2 * v + (1 - beta_2 ) * np.power(g, 2)		# momentum term 2
            	m_hat = m / (1 - beta_1**(p+1) )						# correction term 1
            	v_hat = v / (1 - beta_2**(p+1) )    					# correction term 2
            	w += (-1)* (alpha/np.sqrt(t+1)) * np.divide(m_hat, np.sqrt(v_hat) + epsilon )  		# update


    yield 'smile', w


# In[ ]:


def reducer(key, values):
    # IN :
    # key = None
    # values: list of SVM weight vectors (np.array)
    # OUT:
    # SVM weight vector (np.array)
    # calculated as the average of the SVM weights from values.
    s = values[0]
    T = len(values)
    del values[0]
    for w in values:
        s += w
    w = s / T
    yield w
