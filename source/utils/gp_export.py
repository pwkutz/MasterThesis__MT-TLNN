## decompose multi-fidelity gaussian process models and export them as a dict of variables and matrices into a json file
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def eval_nargp_K(nargp, X_train, X_test = None):
    if X_test is None:
        X_test = X_train
    last_dim = X_train.shape[1] - 1
    r1 = euclidean_distances(X_train[:, last_dim, None], X_test[:, last_dim, None])
    #r1 = tf.maximum(r1, 1e-11)
    K_1 = nargp.kernel.kernels[0].kernels[0].variance.numpy()*pd.DataFrame(np.exp(-0.5*(r1/nargp.kernel.kernels[0].kernels[0].lengthscales.numpy())**2))
    r2 = euclidean_distances(X_train[:, :last_dim], X_test[:, :last_dim])
    #r2 = tf.maximum(r2, 1e-11)
    K_2 = nargp.kernel.kernels[0].kernels[1].variance.numpy()*pd.DataFrame(np.exp(-0.5*(r2/nargp.kernel.kernels[0].kernels[1].lengthscales.numpy())**2))
    K_3 = nargp.kernel.kernels[1].variance.numpy()*pd.DataFrame(np.exp(-0.5*(r2/nargp.kernel.kernels[1].lengthscales.numpy())**2))
    K_combined = K_1*K_2+K_3
    #return K_combined
    return K_combined, K_1, K_2, K_3

def eval_K(model, kernel_type, X_train, X_test = None):
    '''
    Evaluation of different kernels: exponential, squared exponential
    '''
    if X_test is None:
        X_test = X_train
    r = euclidean_distances(X_train, X_test)
    r = tf.maximum(r, 1e-10)

    if kernel_type == "squared_exponential":
        '''
        Evaluating a squared exponential kernel (radial-based; RBF) on given data points
        K = var*exp(-1/2 * (r/lengthscale)^2)
        '''
        K = model.kern.variance[0]*np.exp(-0.5*(r/model.kern.lengthscale[0])**2)
    elif kernel_type == "exponential":
        '''
        Evaluating an exponential kernel on given data points
        K = var*exp((r/lengthscale))
        '''
        K = model.kern.variance[0]*np.exp(-r/model.kern.lengthscale[0])
    return K

def verify_decomp_nargp(X_train, y_train, X_test, nargp):
    Kx = eval_nargp_K(nargp, X_train)[0]
    cholesky_decomp = tf.linalg.cholesky(Kx + np.eye(Kx.shape[0])*1e-11)
    scaled_Y_train = tf.linalg.cholesky_solve(cholesky_decomp, y_train)
    
    Kx_star = eval_nargp_K(nargp, X_train, X_test)[0]

    plt.plot(nargp.predict(X_test[:, :-1])[0], label = "nargp")
    plt.plot(tf.matmul(Kx_star.T, scaled_Y_train), label = "custom")
    plt.legend()
    plt.show()
    return Kx, Kx_star, cholesky_decomp, scaled_Y_train

def verify_decomp_surrogate(kernel_type, X_train, X_test, model):
    Kx_star = eval_K(model, kernel_type, X_train = X_train, X_test = X_test)
    mean, variance = model.predict(X_test)

    plt.plot(mean, label = "surrogate_model")
    plt.plot(np.dot(Kx_star.T, model.posterior.woodbury_vector), label = "custom, with woodbury vec")
    plt.legend()
    plt.show()
    return Kx_star

def export_gp(surrogate, kernel_type, X_train_surrogate, y_train_surrogate, output_index):
    ## surrogate 
    surrogate_predictive_variable = "D:/Vlad/mufintroll/parameters/predictive_variable_surr_" + str(output_index) + "_.csv"
    pd.DataFrame(X_train_surrogate).to_csv(surrogate_predictive_variable, header = False, index = False, sep="\t")

    #Kx = eval_K(model = surrogate, kernel_type=kernel_type, X_train = X_train_surrogate)
    #cholesky_decomp_surr = tf.linalg.cholesky(Kx + np.eye(Kx.shape[0])*1e-6)
    #scaled_Y_train_surr = tf.linalg.cholesky_solve(cholesky_decomp_surr, y_train_surrogate)
    surrogate_woodbury_vector_filename = "D:/Vlad/mufintroll/parameters/woodbury_vector_surr_" + str(output_index) + "_.csv"
    woodbury_vector = surrogate.posterior.woodbury_vector
    pd.DataFrame(woodbury_vector).T.to_csv(surrogate_woodbury_vector_filename, header = False, index = False, sep="\t")

    ### DUMMY VARIABLE JUST FOR SAKE OF NOT CHANGING THE C++ CODE ####
    nargp_predictive_variable = "D:/Vlad/mufintroll/parameters/predictive_variable_nargp_" + str(output_index) + "_.csv"
    pd.DataFrame(np.zeros((10, 10))).to_csv(nargp_predictive_variable, header = False, index = False, sep="\t")

    nargp_scaled_y_filename = "D:/Vlad/mufintroll/parameters/woodbury_vector_nargp_" + str(output_index) + "_.csv"
    pd.DataFrame(np.zeros((10, 10))).T.to_csv(nargp_scaled_y_filename, header = False, index = False, sep="\t")
    #################################################################

    parameters_exprot = {
        # kernel type 
        "kernel_type"               : kernel_type,
        # surrogate parameters
        "variance"                  : surrogate.kern.variance[0],
        "lengthscale"               : surrogate.kern.lengthscale[0],
        # surrogate matrices
        "predictive_variable_file"          : surrogate_predictive_variable,
        "woodbury_vector_file"              : surrogate_woodbury_vector_filename,

        ### DUMMY VARIABLE JUST FOR SAKE OF NOT CHANGING THE C++ CODE
        # NARGP parameters
        "variance_nargp_1"          : 0,
        "lengthscale_nargp_1"       : 0,
        "variance_nargp_2"          : 0,
        "lengthscale_nargp_2"       : 0,
        "variance_nargp_3"          : 0,
        "lengthscale_nargp_3"       : 0,
        # NARGP matrices
        "predictive_variable_nargp_file"    : nargp_predictive_variable,
        "woodbury_vector_nargp_file"        : nargp_scaled_y_filename,
    }

    with open("parameters/parameters_exprot_" + str(output_index) + "_.json", "w") as fp:
        json.dump(parameters_exprot , fp) 

    print("weights are exorted to 'parameters/parameters_exprot_" + str(output_index) + "_.json'")


def export_nargp(surrogate, nargp, kernel_type, X_train_surrogate, X_train_nargp, y_train_nargp, output_index):
    ### decompose and export non-linear auto-regressive GP (no ARD yet! TODO)
    ## surrogate 
    surrogate_predictive_variable = "D:/Vlad/mufintroll/parameters/predictive_variable_surr_" + str(output_index) + "_.csv"
    pd.DataFrame(X_train_surrogate).to_csv(surrogate_predictive_variable, header = False, index = False, sep="\t")
    surrogate_woodbury_vector_filename = "D:/Vlad/mufintroll/parameters/woodbury_vector_surr_" + str(output_index) + "_.csv"
    woodbury_vector = surrogate.posterior.woodbury_vector
    pd.DataFrame(woodbury_vector).T.to_csv(surrogate_woodbury_vector_filename, header = False, index = False, sep="\t")
    ## NARGP
    nargp_predictive_variable = "D:/Vlad/mufintroll/parameters/predictive_variable_nargp_" + str(output_index) + "_.csv"
    pd.DataFrame(X_train_nargp).to_csv(nargp_predictive_variable, header = False, index = False, sep="\t")
    Kx = eval_nargp_K(nargp = nargp, X_train = X_train_nargp)[0]
    cholesky_decomp = tf.linalg.cholesky(Kx + np.eye(Kx.shape[0])*1e-6)
    scaled_Y_train = tf.linalg.cholesky_solve(cholesky_decomp, y_train_nargp)
    nargp_scaled_y_filename = "D:/Vlad/mufintroll/parameters/woodbury_vector_nargp_" + str(output_index) + "_.csv"
    pd.DataFrame(scaled_Y_train).T.to_csv(nargp_scaled_y_filename, header = False, index = False, sep="\t")

    parameters_exprot = {
        # kernel type 
        "kernel_type"               : kernel_type,
        # surrogate parameters
        "variance"                  : surrogate.kern.variance[0],
        "lengthscale"               : surrogate.kern.lengthscale[0],
        # NARGP parameters
        "variance_nargp_1"          : float(nargp.kernel.kernels[0].kernels[0].variance.numpy()),
        "lengthscale_nargp_1"       : float(nargp.kernel.kernels[0].kernels[0].lengthscales.numpy()),
        "variance_nargp_2"          : float(nargp.kernel.kernels[0].kernels[1].variance.numpy()),
        "lengthscale_nargp_2"       : float(nargp.kernel.kernels[0].kernels[1].lengthscales.numpy()),
        "variance_nargp_3"          : float(nargp.kernel.kernels[1].variance.numpy()),
        "lengthscale_nargp_3"       : float(nargp.kernel.kernels[1].lengthscales.numpy()),
        # surrogate matrices
        "predictive_variable_file"          : surrogate_predictive_variable,
        "woodbury_vector_file"              : surrogate_woodbury_vector_filename,
        # NARGP matrices
        "predictive_variable_nargp_file"    : nargp_predictive_variable,
        "woodbury_vector_nargp_file"        : nargp_scaled_y_filename,
    }

    with open("parameters/parameters_exprot_" + str(output_index) + "_.json", "w") as fp:
        json.dump(parameters_exprot , fp) 

    print("weights are exorted to 'parameters/parameters_exprot_" + str(output_index) + "_.json'")

def nargp_pred(X_train, X_test, nargp, scaled_Y_train):
    #Kx = eval_nargp_K(nargp, X_train = X_train)
    Kx_star = eval_nargp_K(nargp, X_train = X_train, X_test = X_test)[0]
    #cholesky_decomp = tf.linalg.cholesky(Kx + np.eye(Kx.shape[0])*1e-11)
    #scaled_Y_train = tf.linalg.cholesky_solve(cholesky_decomp, y_train)
    
    return tf.matmul(Kx_star.T, scaled_Y_train)