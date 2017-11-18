import numpy as np

def transform(X):
	# Uses Random Fourier Features based on Radial Basis Function Kernel
	
    # Make sure this function works for both 1D and 2D NumPy arrays.
    
    if np.ndim(X) == 1:
        X = np.reshape(X, (1, len(X) ) )
    
    # dimensions of training set
    n, d = X.shape
    # dimension of RFF features
    m = 20000 #9600
    # parameter for the normal distribution (co)variance
    gamma =  17.625 # 17.625 
    
    # set random seed to ensure the same transform for every input
    np.random.seed(1)
    # sample from the correct distributions
    omega = np.sqrt(2*gamma)*np.random.standard_normal((d, m) )
    b = np.random.random_sample(m) * 2 * np.pi
    
    # compute transformation
    Z = np.sqrt(2 / float(m)) * np.cos(np.add(np.dot(X, omega), b ) )
    return Z

def mapper(key, value):
    
    data = []
    for val in value:
        data.append(np.array(val.split(), dtype=float ) )
    
    data = np.array(data)
    # number of training examples
    n = data.shape[0]
    
    # Training Examples: 
    y = data[:, 0]        # class labels
    x = data[:, 1 : ]     # features
    x = transform(x)	  # feature tranformation
    d = x.shape[1]        # feature dimension


    
    # Try momentum Gradient Descent
    z = w = np.zeros(d)   	# SVM vector initialization
    alpha = 0.3		  		# stepsize parameter 0.4
    beta = 0.9			  	# momentum 0.7
    
    # passes through the data
    for i in range(1):
    	# permutation for SGD
    	perm = np.random.permutation(n)
    	for t in perm:
            if y[t]*np.dot(x[t, :], w ) < 1 :
                z = beta * z - y[t] * x[t, :] 		# momentum term
                w += - alpha * z 					# update
            else:
            	z = beta * z
            	w += - alpha * z
    
    yield 'smile', w

def reducer(key, values):
    # IN :
    # key = None
    # values: list of SVM weight vectors (np.array)
    # OUT: 
    # SVM weight vector (np.array)
    # calculated as the average of the SVM weights from values.
    
    w = np.mean(values, axis = 0)
    
    yield w