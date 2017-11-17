import numpy as np

def transform(X):
	# Uses Random Fourier Features based on Radial Basis Function Kernel
	
    # Make sure this function works for both 1D and 2D NumPy arrays.
    
    # if X is 1D convert it to 2D
    if np.ndim(X) == 1:
        X = np.reshape(X, (1, len(X) ) )
    
    # dimensions of training set
    n, d = X.shape
    # dimension of RFF features
    m = 9600
    # parameter for the normal distribution (co)variance
    gamma = 17.625
    
    # set random seed to ensure the same transform for every input
    np.random.seed(1)
    # sample from the correct distributions
    omega = np.sqrt(2*gamma)*np.random.standard_normal((d, m) )
    b = np.random.random_sample(m) * 2 * np.pi
    
    # compute transformation
    Z = np.sqrt(2 / float(m)) * np.cos(np.add(np.dot(X, omega), b ) )
    
    return Z

def mapper(key, value):
    
    # get the data input as an np.array
    data = []
    for val in value:
        data.append(np.array(val.split(), dtype=float ) )
    data = np.array(data)
    
    # number of training examples
    n = data.shape[0]
        
    # Training Examples: 
    y = data[:, 0]        # class labels
    x = data[:, 1 : ]     # features
    z = transform(x)	  # transformation
    d = z.shape[1]        # feature dimension
    
    # Try ADAM
    m = v = w = np.zeros(d)   	# SVM vector initialization
    alpha = 1					# stepsize parameter
    beta_1 = 0.9				# momentum param
    beta_2 = 0.999				# momentum param
    epsilon = 1e-8				# security param
    
    # number passes through the data
    for iter in range(15):
    	# permutation for SGD
    	perm = np.random.permutation(n)
    	# ADAM pass through the data
    	for p, t in enumerate(perm):
            p *= 1.5

            if y[t]*np.dot(z[t, :], w ) < 1 :								# if gradient is nonzero
            	g = (-1)*y[t] * z[t, :]
            	m = beta_1 * m + (1 - beta_1 ) * g							# momentum term 1   
            	v = beta_2 * v + (1 - beta_2 ) * np.power(g, 2)				# momentum term 2
            	m_hat = m / (1 - beta_1**(iter+p+1) )						# correction term 1
            	v_hat = v / (1 - beta_2**(iter+p+1) )    					# correction term 2
            	w += (-1)* (alpha/np.sqrt(iter+p+1)) * np.divide(m_hat, np.sqrt(v_hat) + epsilon )  		# update
            else:															# if gradient is zero, ADAM still does a little
            	m = beta_1 * m
            	v = beta_2 * v
            	m_hat = m / (1 - beta_1**(iter+p+1) )
            	v_hat = v / (1 - beta_2**(iter+p+1) )
            	w += (-1) * (alpha/np.sqrt(iter+p+1)) * np.divide(m_hat, np.sqrt(v_hat) + epsilon )
            	
           
    yield 'smile', w

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
