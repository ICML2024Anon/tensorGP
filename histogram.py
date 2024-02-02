import TN_mutable
from utils import kernel, find_max_TT_rank, hilbert_gp_gaussian_prior
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.constants import golden

# Plotting
n_fig_per_row = 4
textwidth_pt = 487.8225
points_per_inch = 72.27
textwidth_inches = textwidth_pt / points_per_inch
image_size = textwidth_inches / n_fig_per_row

N = 1    # this is NOT the number of data points
D = 16
M = 10
lengthscale = 1
prior_cov = np.identity(M)
prior_cov = hilbert_gp_gaussian_prior(M,lengthscale).eval()
sqrt_prior_covariance = np.sqrt(prior_cov)

n_samples = 10000
n_bins = int(np.ceil(np.sqrt(n_samples)))




np.random.seed(0)
X = np.random.normal(0,1, size=(N,D))
y = np.random.normal(0,1, size=(N,))   # placeholder, needed by PyMC
# Compute sqrt of kernel at random point X
k = kernel(X,X,prior_cov).flatten()
k = k[0]
normal_CDF = lambda x: stats.norm.cdf(x,loc=0,scale=np.sqrt(k))
normal_PDF = lambda x: stats.norm.pdf(x,loc=0,scale=np.sqrt(k))


R_CPD = np.array([1,2,5,10,20,50,100,1000,10000])
parameters = R_CPD*M*D

for R in R_CPD:
    cpd_model = TN_mutable.CPD(X,y,sqrt_prior_covariance,R,1)
    cpd_samples = cpd_model.sample_prior(n_samples=n_samples,random_seed=0).values.flatten()
    P = cpd_model.P
    filename = "hist_CPD_"+str(D)+"_"+str(P)
    x = np.linspace(-2*np.sqrt(k),2*np.sqrt(k),n_samples)
    plt.figure(figsize=(image_size, image_size))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.xlim(-2*np.sqrt(k),2*np.sqrt(k))
    plt.ylim(0,golden*normal_PDF(0))
    plt.hist(cpd_samples, bins=n_bins,range=(-2*np.sqrt(k),2*np.sqrt(k)),density=True)
    plt.plot(x,normal_PDF(x),'k-',linewidth=0.5)
    plt.savefig(filename+".pdf")
    plt.close()


print("TT")
for P in parameters:
    R, n_iter_TT = find_max_TT_rank(P,D,M)
    tt_model = TN_mutable.TT(X,y,sqrt_prior_covariance,R,1)
    P_TT = tt_model.P
    tt_samples = tt_model.sample_prior(n_samples=n_samples,random_seed=0).values.flatten()
    filename = "hist_TT_"+str(D)+"_"+str(P_TT)
    x = np.linspace(-2*np.sqrt(k),2*np.sqrt(k),n_samples)
    plt.figure(figsize=(image_size, image_size))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.xlim(-2*np.sqrt(k),2*np.sqrt(k))
    plt.ylim(0,golden*normal_PDF(0))
    plt.hist(tt_samples, bins=n_bins,range=(-2*np.sqrt(k),2*np.sqrt(k)),density=True,color='orange')
    plt.plot(x,normal_PDF(x),'k-',linewidth=0.5)
    plt.savefig(filename+".pdf")
    plt.close()