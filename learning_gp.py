import numpy as np
import TN_mutable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from utils import kernel, hilbert_gp_gaussian_prior, find_max_TT_rank, hilbert_GP_RBF


file_names = ["yacht","energy"]
folder = ...

# Loop over filenames

for file_name in file_names:
    file_path = folder+file_name
    print("Processing: "+file_name)
    X = np.loadtxt(file_path+".csv", delimiter=';')
    y = X[:,-1]
    X = X[:,:-1]


    # Preprocess
    X  = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1))

    # Select the number of basis functions
    M = 10
    # Select number of samples
    n_chains = 4
    n_samples = 1000
    tune = 1000

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # GP
    custom_kernel = ConstantKernel()*hilbert_GP_RBF(M=M,length_scale=1.0) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=custom_kernel, n_restarts_optimizer=10,alpha=0.0,random_state=0)
    gp.fit(X_train, y_train)
    gp_predictions = gp.predict(X_test)
    mean_RMSE_GP = mean_squared_error(y_test,gp_predictions)
    
    print("GP RMSE: "+str(mean_RMSE_GP))

    # Hyperparameters
    hyperparameters = gp.kernel_.get_params()
    scaling = hyperparameters['k1__k1__constant_value']
    lengthscale = hyperparameters['k1__k2__length_scale']
    sigma_noise = hyperparameters['k2__noise_level']

    # Parameters
    N,D = np.shape(X_train)
    R_CPD = np.array([1,2,5,10,20],dtype=np.int64)
    parameters = M*D*R_CPD

    # Define prior with given hyperparameters
    sqrt_prior_covariance = np.sqrt(hilbert_gp_gaussian_prior(M,lengthscale).eval())

    # Initialize containers
    mean_RMSE_CPD = np.zeros(parameters.shape)
    std_RMSE_CPD = np.zeros(parameters.shape)
    P_CPD = np.zeros(parameters.shape)

    mean_RMSE_TT = np.zeros(parameters.shape)
    std_RMSE_TT = np.zeros(parameters.shape)
    P_TT = np.zeros(parameters.shape)

    # CPD
    print("CPD")
    for idx in range(parameters.shape[0]):
        R = R_CPD[idx]
        cpd_model = TN_mutable.CPD(X_train,y_train.flatten(),sqrt_prior_covariance,R,sigma_noise)
        P_CPD[idx] = cpd_model.P
        cpd_model.fit(random_seed=0,n_samples=n_samples,tune=tune,n_chains=n_chains) 
        cpd_predictions = cpd_model.predict(X_test,random_seed=0).predictions.to_array().values
        cpd_predictions = cpd_predictions[0,:,:,:] # get rid of singleton
        cpd_samples = cpd_predictions
        cpd_predictions = np.mean(cpd_predictions,axis=1) # posterior mean
        cpd_predictions = cpd_predictions.T # N_test x N_chains
        mean_RMSE_CPD[idx] = np.mean(np.sqrt(np.mean(np.power(y_test-cpd_predictions,2),axis=0)))
        std_RMSE_CPD[idx] = np.std(np.sqrt(np.mean(np.power(y_test-cpd_predictions,2),axis=0)))
        np.savez(file_name+".npz",cpd_samples=cpd_samples,mean_RMSE_GP=mean_RMSE_GP,mean_RMSE_CPD=mean_RMSE_CPD,std_RMSE_CPD=std_RMSE_CPD,P_CPD=P_CPD,mean_RMSE_TT=mean_RMSE_TT,std_RMSE_TT=std_RMSE_TT,P_TT=P_TT)
        print("CPD rank: "+str(R)+" RMSE: "+str(mean_RMSE_CPD[idx])+" STD: "+str(std_RMSE_CPD[idx]))

    # TT
    print("TT")
    for idx in range(parameters.shape[0]):
        R, n_iter_TT = find_max_TT_rank(parameters[idx],D,M)
        tt_model = TN_mutable.TT(X_train,y_train.flatten(),sqrt_prior_covariance,R,sigma_noise)
        P_TT[idx] = tt_model.P
        tt_model.fit(random_seed=0,n_samples=n_samples,tune=tune,n_chains=n_chains)
        tt_predictions = tt_model.predict(X_test,random_seed=0).predictions.to_array().values
        tt_predictions = tt_predictions[0,:,:,:] # get rid of singleton
        tt_predictions = np.mean(tt_predictions,axis=1) # posterior mean
        tt_predictions = tt_predictions.T # N_test x N_chains        mean_RMSE_TT[idx] = np.mean(np.sqrt(np.mean(np.power(y_test-tt_predictions,2),axis=0)))
        mean_RMSE_TT[idx] = np.mean(np.sqrt(np.mean(np.power(y_test-tt_predictions,2),axis=0)))
        std_RMSE_TT[idx] = np.std(np.sqrt(np.mean(np.power(y_test-tt_predictions,2),axis=0)))
        np.savez(file_name+".npz",cpd_samples=cpd_samples,mean_RMSE_GP=mean_RMSE_GP,mean_RMSE_CPD=mean_RMSE_CPD,std_RMSE_CPD=std_RMSE_CPD,P_CPD=P_CPD,mean_RMSE_TT=mean_RMSE_TT,std_RMSE_TT=std_RMSE_TT,P_TT=P_TT)
        print("TT rank: "+str(R)+" RMSE: "+str(mean_RMSE_TT[idx])+" STD: "+str(std_RMSE_TT[idx]))