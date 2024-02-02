import numpy as np
import pymc as pm
import pytensor.tensor as pt


def khatri_rao(L,R):
    r1,c1 = L.shape   
    r2,c2 = R.shape
    #y=repmat(L,1,c2).*kron(R, ones(1, c1));  
    return pt.tile(L,(1,c2))*pt.slinalg.kron(R, pt.ones((1,c1)))

def hilbert_gp_features(x,L,M):
    idx = pt.arange(1,M+1,1)
    matrix  = pt.tile(idx, (x.shape[0],1))
    a = pt.pi*(L+x[:, pt.newaxis])*matrix
    return pt.sin(a/2/L)/pt.sqrt(L)

def feature(x,M):
    return hilbert_gp_features(x,3,M)

def hilbert_gp_gaussian_prior(M,lengthscale):
    # w = 1:M;
    # S = sqrt(2*pi)*lengthscale*exp(-(pi*w/2).^2*lengthscale^2/2);
    w = pt.arange(1,M+1,1)
    return pt.diag( pt.sqrt(2*pt.pi)*lengthscale*pt.exp(-pt.power(pt.pi*w/2,2)*pt.power(lengthscale,2)/2))

def kernel(x,z,prior_cov):
    x = np.atleast_2d(x)
    z = np.atleast_2d(z)
    D = np.shape(x)[1]
    M = np.shape(prior_cov)[0]
    k = 1
    for d in range(D):
        a = feature(x[:,d],M).eval()
        b = feature(z[:,d],M).eval()
        k *= np.inner(a, np.matmul( prior_cov,  b.T).T)
    return k


def find_max_TT_rank(target_P,D,M):
    new_array = np.ones((D+1,),dtype=np.int64)
    P = M* np.sum(new_array[:-1] * new_array[1:])
    number_iterations = 0
    while P < target_P:
        for idx in range(1,D,1):
            new_array[idx] += 1
            number_iterations += 1
            P = M* np.sum(new_array[:-1] * new_array[1:])
    return new_array, number_iterations


from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
class hilbert_GP_RBF(Kernel):
    def __init__(self, M, length_scale, length_scale_bounds=(1e-2,2)):
        # Initialize the parameters of your custom kernel
        self.M = M
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        # Call the superclass constructor
        super().__init__()
        
    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)
    
    def __call__(self, X, Y=None, eval_gradient=False):
        # The gradient of the kernel k(X, X) with respect to the log of the
        # hyperparameter of the kernel. Only returned when `eval_gradient`
        # is True.

        if Y is None:
            Y = X
        prior_covariance = hilbert_gp_gaussian_prior(self.M,self.length_scale).eval()
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        D = np.shape(X)[1]
        covariance_matrix = 1
        grad_matrix = 0
        w = np.arange(1,self.M+1,1)
        for d in range(D):
            a = feature(X[:,d],self.M).eval()
            b = feature(Y[:,d],self.M).eval()
            covariance_matrix_d = np.inner(a, np.matmul( prior_covariance,  b.T).T)
            covariance_matrix *= covariance_matrix_d

            if eval_gradient:
                # Compute the gradient of the kernel if eval_gradient is True
                temp = np.power(np.pi*w*np.exp(self.length_scale),2)
                grad_theta = np.sqrt(2*np.pi)*np.exp(self.length_scale-1/8*temp)*(1-1/4*temp)
                grad_matrix_d = np.inner(a, np.matmul( np.diag(grad_theta),  b.T).T)
                grad_matrix += np.divide(grad_matrix_d,covariance_matrix_d)

        if eval_gradient:
            grad_matrix = grad_matrix[:,:,np.newaxis]
            return covariance_matrix, grad_matrix
        else:
            return covariance_matrix

    def diag(self, X):
        return np.diag(self.__call__(X, Y=X, eval_gradient=False))
    
    def is_stationary(self):
        return True

    def __repr__(self):
        return "{0}(length_scale={1:.3g})".format(
            self.__class__.__name__, np.ravel(self.length_scale)[0])