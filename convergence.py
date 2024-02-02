import TN_mutable
import numpy as np
from scipy import stats
from utils import kernel, find_max_TT_rank, hilbert_gp_gaussian_prior

# Plotting

n_fig_per_row = 4
textwidth_pt = 487.8225
points_per_inch = 72.27
textwidth_inches = textwidth_pt / points_per_inch
image_size = textwidth_inches / n_fig_per_row

N = 1    # this is NOT the number of data points
D = 2
M = 10
lengthscale = 1
prior_cov = np.identity(M)
prior_cov = hilbert_gp_gaussian_prior(M,lengthscale).eval()
sqrt_prior_covariance = np.sqrt(prior_cov)

n_restarts = 10
n_samples = 10000
n_bins = int(np.ceil(np.sqrt(n_samples)))

max_parameters = 10000
max_R_CPD = np.ceil(max_parameters/M/D).astype(int)
max_R_TT, n_iter_TT = find_max_TT_rank(max_parameters,D,M)

# Initialize containers
ks_CPD = np.zeros((max_R_CPD,n_restarts))
P_CPD = np.zeros((max_R_CPD,))

ks_TT = np.zeros((n_iter_TT,n_restarts))
P_TT =np.zeros((n_iter_TT,))




for n in range(n_restarts):
    print("restart "+str(n))
    np.random.seed(n)
    X = np.random.normal(0,1, size=(N,D))
    y = np.random.normal(0,1, size=(N,))   # placeholder, needed by PyMC
    # Compute sqrt of kernel at random point X
    k = kernel(X,X,prior_cov).flatten()
    k = k[0]
    normal_CDF = lambda x: stats.norm.cdf(x,loc=0,scale=np.sqrt(k))
    normal_PDF = lambda x: stats.norm.pdf(x,loc=0,scale=np.sqrt(k))

    print("CPD")
    for idx in range(max_R_CPD):
        R = idx+1
        cpd_model = TN_mutable.CPD(X,y,sqrt_prior_covariance,R,1)
        P_CPD[idx] = cpd_model.P
        cpd_samples = cpd_model.sample_prior(n_samples=n_samples,random_seed=n).values.flatten()
        ks_test = stats.cramervonmises(cpd_samples, normal_CDF)
        ks_CPD[idx,n] = ks_test.statistic

    print("TT")
    R = np.ones((D+1,),dtype=np.int64)
    idx_ks = 0
    plot_idx = 0
    while not np.array_equal(R,max_R_TT):
        for idx in range(1,D,1):
            tt_model = TN_mutable.TT(X,y,sqrt_prior_covariance,R,1)
            P_TT[idx_ks] = tt_model.P
            tt_samples = tt_model.sample_prior(n_samples=n_samples,random_seed=n).values.flatten()
            ks_test = stats.cramervonmises(tt_samples, normal_CDF)
            ks_TT[idx_ks,n] = ks_test.statistic
            R[idx] += 1
            idx_ks += 1

    np.savez("convergence_D_"+str(D)+".npz",ks_CPD=ks_CPD,ks_TT=ks_TT,P_CPD=P_CPD,P_TT=P_TT)