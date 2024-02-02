import pymc as pm
import numpy as np
import pytensor.tensor as pt
from utils import feature, khatri_rao

class CPD:
    def __init__(self,X,y,sqrt_prior_covariance,R,sigma_noise):
        self.y = y
        self.sqrt_prior_covariance = sqrt_prior_covariance
        self.R = R
        self.sigma_noise = sigma_noise
        N = np.shape(X)[0]
        self.D = np.shape(X)[1]
        self.M = np.shape(sqrt_prior_covariance)[0]
        self.P = self.M*self.R*self.D
        self.CPD_model = pm.Model()

        with self.CPD_model:
            X = pm.MutableData ('X', X)
            # Define prior
            factors = np.empty(self.D, dtype=object)
            for d in range(self.D):
                factors[d] = (sqrt_prior_covariance / pt.power(self.R,1/(2*self.D))) @ pm.Normal(f'factor_{d}',mu=0, sigma = 1, shape = (self.M,self.R))

            # Model output
            output = pm.math.ones((X.shape[0],self.R))
            for d in range(self.D):
                a = pm.math.matmul(feature(X[:,d],self.M),factors[d])
                output = output*a
            output = pm.Deterministic('CPD_output',pm.math.sum(output,axis=1))

            # Likelihood (observed data)
            likelihood = pm.Normal('y', mu=output, sigma=self.sigma_noise, observed=self.y,shape=output.shape)

    def sample_prior(self,random_seed,n_samples=1000):
        with self.CPD_model:
            samples = pm.sample_prior_predictive(samples=n_samples,random_seed=random_seed,var_names=['CPD_output']).prior.CPD_output
        return samples
    
    def fit(self,random_seed,n_samples=1000,tune=1000,n_chains=1):
        with self.CPD_model:
            self.trace = pm.sample(draws=n_samples,random_seed=random_seed, tune=tune, progressbar=False, cores=1, chains=n_chains)

    def predict(self,X_test,random_seed):
        with self.CPD_model:
            pm.set_data({"X": X_test})
            return pm.sample_posterior_predictive(self.trace,random_seed=random_seed,predictions=True,var_names=['CPD_output'],progressbar=False)


class TT:
    def __init__(self,X,y,sqrt_prior_covariance,R,sigma_noise):
        self.X = X
        self.y = y
        self.sqrt_prior_covariance = sqrt_prior_covariance
        self.R = R
        self.sigma_noise = sigma_noise
        self.N = np.shape(X)[0]
        self.D = np.shape(X)[1]
        self.M = np.shape(sqrt_prior_covariance)[0]
        self.P = self.M* np.sum(self.R[:-1] * self.R[1:])
        self.TT_model = pm.Model()

        with self.TT_model:
            X = pm.MutableData ('X', X)
            # Define prior
            cores = np.empty(self.D, dtype=object)
            for d in range(self.D):
                cores[d] = (sqrt_prior_covariance / pt.sqrt(pt.sqrt(R[d]*R[d+1]))) @ pm.Normal(f'factor_{d}',mu=0, sigma = 1, shape = (self.M,self.R[d]*self.R[d+1]))

            output = pm.math.matmul(feature(X[:,0],self.M),cores[0])  # N x R
            for d in range(1,self.D):
                temp = khatri_rao(output,feature(X[:,d],self.M))      # N x MR
                output = temp @ cores[d].reshape((self.M*self.R[d],self.R[d+1]))
            output = pm.Deterministic('TT_output',output.flatten())

            # Likelihood (observed data)
            likelihood = pm.Normal('likelihood', mu=output, sigma=self.sigma_noise, observed=self.y,shape=output.shape)

    def sample_prior(self,random_seed,n_samples=1000):
        with self.TT_model:
            samples = pm.sample_prior_predictive(samples=n_samples,random_seed=random_seed,var_names=['TT_output']).prior.TT_output
        return samples
    
    def fit(self,random_seed,n_samples=1000,tune=1000,n_chains=1):
        with self.TT_model:
            self.trace = pm.sample(draws=n_samples,random_seed=random_seed, tune=tune, progressbar=False, cores=1, chains=n_chains)

    def predict(self,X_test,random_seed):
        with self.TT_model:
            pm.set_data({"X": X_test})
            return pm.sample_posterior_predictive(self.trace,random_seed=random_seed,predictions=True,var_names=['TT_output'], progressbar=False)