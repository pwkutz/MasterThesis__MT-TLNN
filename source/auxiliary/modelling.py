import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
import GPy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from sklearn.metrics.pairwise import euclidean_distances
import nn
import tensorflow as tf
from tensorflow.keras.models import load_model
from time import time
import pickle

def load_data(num_medium_fidelity, num_test_points):
    terra = pd.read_csv("terra_dataset_refined.csv", header=None)
    scm = pd.read_csv("scm_dataset_refined.csv", header=0, index_col= 0)
    #input_features = ["coordinate_x","coordinate_z","velocity_x","velocity_y","velocity_z","angular_velocity_y","gravity_x","gravity_z"]
    #predicted_feature = "force_x"
    rg = np.random.default_rng(12345)
    scm_dwnsmpl_idx = rg.integers(low=0, high=scm.shape[0], size=num_medium_fidelity)
    #trr_dwnsmpl_inx = rg.integers(low=0, high=terra.shape[0], size=num_low_fidelity)
    test_dwnsmpl_inx = rg.integers(low=0, high=scm.shape[0], size=num_test_points)

    #X_low = np.array(terra.iloc[:, 6:18])[trr_dwnsmpl_inx]
    X_medium = np.array(scm.iloc[:, 6:18])[scm_dwnsmpl_idx, :]
    #X_low_surf = np.array(terra.iloc[:, 18:])[trr_dwnsmpl_inx]
    X_medium_surf = np.array(scm.iloc[:, 18:])[scm_dwnsmpl_idx, :]

    #Y_low = np.array(terra.iloc[:, 0])[:,None][trr_dwnsmpl_inx]
    Y_low_as_input = np.array(terra.iloc[:, 0])[:,None][scm_dwnsmpl_idx]
    Y_medium = np.array(scm.iloc[:, 0])[:,None][scm_dwnsmpl_idx, :]
    Y_delta_medium = np.array(scm.iloc[:, 0])[:,None][scm_dwnsmpl_idx, :] - np.array(terra.iloc[:, 0])[:,None][scm_dwnsmpl_idx, :]
    
    X_test = np.array(scm.iloc[:, 6:18])[100:400]
    X_test_surf = np.array(scm.iloc[:, 18:])[100:400]
    Y_test = np.array(scm.iloc[:, 0])[:,None][100:400]
    Y_low_as_input_test = np.array(terra.iloc[:, 0])[:,None][100:400]

    return X_medium, X_medium_surf, Y_medium, X_test, X_test_surf, Y_test, Y_delta_medium, Y_low_as_input, Y_low_as_input_test

def train_gpr(X, y, dim):
    kernel = GPy.kern.RBF(dim, ARD = True)
    model = GPy.models.GPRegression(X=X, Y=y, kernel=kernel)
    model[".*Gaussian_noise"] = model.Y.var()*0.0001 # the lower coeff is (0.1 currently), the more I "belive" training Y 
    # unclear - do we need this, or not?
    model[".*Gaussian_noise"].fix()
    model.optimize(max_iters = 1000)
    model[".*Gaussian_noise"].unfix()
    model[".*Gaussian_noise"].constrain_positive()
    model.optimize_restarts(3, optimizer = "lbfgs",  max_iters = 1000)
    return model, kernel

def train_mf_gpr(Y_low_of_X_medium, X_medium, y_medium, nn_train, input_dim, active_dimensions):
    stacked_X = np.hstack((X_medium, Y_low_of_X_medium, nn_train))
    k2 = GPy.kern.RBF(1, active_dims = [input_dim])*GPy.kern.RBF(input_dim, active_dims = active_dimensions) \
        + GPy.kern.RBF(input_dim, active_dims = active_dimensions)
    m2 = GPy.models.GPRegression(X=stacked_X, Y=y_medium, kernel=k2)
    m2[".*Gaussian_noise"] = m2.Y.var()*0.0001 # 1 - max uncertainty in the training data points, 0 - total certainty
    #m2[".*Gaussian_noise"].fix()
    m2.optimize(max_iters = 1000)
    #m2[".*Gaussian_noise"].unfix()
    #m2[".*Gaussian_noise"].constrain_positive()
    m2.optimize_restarts(3, optimizer = "lbfgs",  max_iters = 1000)

    return m2, k2

def train_mf_bnn(X, y_medium, X_test, y_test, learning_rate = 0.001, hidden_units = [8, 8]):
    number_of_features = X.shape[1]
    train_size = X.shape[0]

    bnn_model_mf = nn.BNN(train_size, hidden_units, number_of_features)
    bnn_model_mf.create_bnn_model()
    bnn_model_mf.compile(
        optimizer_ =tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss_ = tf.keras.losses.MeanSquaredError(),
        metrics_ =[tf.keras.metrics.RootMeanSquaredError()],
    )

    bnn_model_mf.fit(X, y_medium, num_epochs=5, X_test=X_test, y_test=y_test)
    return(bnn_model_mf)

def metrics(model, X_test, y_test, plots_name):
    mu, var = model.predict(X_test)
    var = abs(var)
    print([mean_squared_error(y_test, mu), mean_absolute_error(y_test, mu)])
    plt.rcParams["figure.figsize"] = (16,8)
    plt.plot(y_test, color='b', label = "SCM simulation")
    plt.plot(mu, color='k', label = plots_name)
    plt.fill_between(x = list(range(len(mu))),
                    y1 = (mu - 2 * np.sqrt(var)).ravel(),
                    y2 = (mu + 2 * np.sqrt(var)).ravel(),
                    alpha=.3, fc='black', ec='None', label='95% confidence interval')

    plt.legend(loc='lower left')
    plt.xlabel("Timestamp, 0.1 sec")
    plt.ylabel("Traction force, N")
    plt.show()
    ### COP - confidence of prediction

def metrics_hybrid(model, X_test, y_low, y_test, plots_name):
    mu, var = model.predict(X_test)
    var = abs(var)
    print([mean_squared_error(y_test, mu+y_low), mean_absolute_error(y_test, mu+y_low)])
    plt.rcParams["figure.figsize"] = (16,8)
    plt.plot(y_test, color='b', label = "SCM simulation")
    plt.plot(mu+y_low, color='k', label = plots_name)
    plt.fill_between(x = list(range(len(mu))),
                    y1 = (mu+y_low - 2 * np.sqrt(var)).ravel(),
                    y2 = (mu+y_low + 2 * np.sqrt(var)).ravel(),
                    alpha=.3, fc='black', ec='None', label='95% confidence interval')

    plt.legend(loc='lower left')
    plt.xlabel("Timestamp, 0.1 sec")
    plt.ylabel("Traction force, N")
    plt.show()
    #return([mean_squared_error(y_test, mu+y_low.transpose()[0]), mean_absolute_error(y_test, mu+y_low.transpose()[0])])
    ### COP - confidence of prediction

def save_weights(path, model):
    # write relevant matrices
    narray = pd.DataFrame(model._predictive_variable)
    narray.to_csv(os.path.join(path, "predictive_variable.csv"), header = False, index = False, sep="\t")
    woodbury_vector = pd.DataFrame(model.posterior.woodbury_vector.T)
    woodbury_vector.to_csv(os.path.join(path, "woodbury_vector.csv"), header = False, index = False, sep="\t")

'''
def assertion():
    # example of prediction on one element from the test 
    distances_matrix_test = euclidean_distances(X_medium, X_test)
    k_X_star_X = (k.variance[0])*np.exp((-0.5)*(distances_matrix_test**2)/(k.lengthscale[0]**2))
    posterior = m.inference_method.inference(m.kern, m.X, m.likelihood, m.Y)[0]

    Kx = m.kern.K(m._predictive_variable, X_test)
    assert(m.kern.variance[0]*pd.DataFrame(np.exp(-0.5*(euclidean_distances(m._predictive_variable, X_test)/m.kern.lengthscale[0])**2)) == Kx) # maybe not exactly the same...
    # woodbury matrix identiry says that the inverse of k-ranking correction of the original matrix could be done by adding inverse update to the inverse of original matrix
    # (A + UCV)^-1 = A^-1 + (A^-1)U((C^-1+V(A^-1)U)^-1)^-1(V)A^-1
    # k-ranking correction occures when new (observation) data is recieved and initial matrix should be updated 
    
    ### assertion #2 - should be the same for the X_test
    Kx_ = m.kern.variance[0]*pd.DataFrame(np.exp(-0.5*(euclidean_distances(m._predictive_variable, X_test)/m.kern.lengthscale[0])**2))
    assert(m.predict(X_test)[0] == np.dot(Kx_.T, m.posterior.woodbury_vector))
    
    #plt.plot(np.dot(Kx.T, m.posterior.woodbury_vector))
    # plt.plot(np.dot(Kx_.T, m.posterior.woodbury_vector))
    # plt.plot(m.predict(X_test)[0])

    ######## assertion number 3 - results for np.zeros((1, 12)) should be the same, in case of the vector, not matrix #####
    Kx_ = m.kern.variance[0]*pd.DataFrame(np.exp(-0.5*(euclidean_distances(m._predictive_variable, np.zeros((1, 12)))/m.kern.lengthscale[0])**2))
    m.predict(np.zeros((1, 12)))[0] == np.dot(Kx_.T, m.posterior.woodbury_vector)
'''

### automatic kernel determination 
## kernel flows? 
def kernel_flow():
    kernel = None
    ### please insert your code here, lol :D 
    return kernel

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    points_low_fidelity = 1000
    point_medium_fid = 5
    point_test_medium_fid = 300
    X_medium, X_medium_surf, y_medium, X_test, X_test_surf, y_test, y_delta_medium, y_low_input, y_low_as_input_test = load_data(point_medium_fid, point_test_medium_fid)

    input_dim = 12
    active_dimensions = np.arange(0,input_dim)
    model_terra = load_model('C://Users//fedi_vl//Documents//work//terra.h5')
    #model_scm = load_model('C://Users//fedi_vl//Documents//work//scm.h5')
    #x_trans_low = nn.transform_to_scans(pd.DataFrame(X_low_surf))
    x_trans_medium = nn.transform_to_scans(pd.DataFrame(X_medium_surf))
    X_test_trans = nn.transform_to_scans(pd.DataFrame(X_test_surf)) 
    nn_train = model_terra.predict(x_trans_medium)
    nn_test = model_terra.predict(X_test_trans)
    print("BASELINE, MSE BETWEEN SCM ANT TERRA: ", mean_absolute_error(y_test, y_low_as_input_test))
    # simple one-fidelity model
    print("Gaussian process, single fidelity")
    start = time()
    print(X_medium.shape)
    print(y_medium.shape)
    m, k = train_gpr(X_medium, y_medium, input_dim)
    metrics(m, X_test, y_test, "Gaussian process, single fidelity")
    end = time()
    print("Elapsed time: ", end-start)
    #with open("medium_single_fidelity_GP.dump" , "wb") as f:
    #    pickle.dump(m, f)  
    
    # mf model
    print("Gaussian process, multi fidelity")
    start = time()
    m_mf, k_mf = train_mf_gpr(y_low_input, X_medium, y_medium, nn_train, input_dim+1, np.arange(0,input_dim+1))
    metrics(m_mf, np.hstack((np.array(X_test), y_low_as_input_test, nn_test)), y_test, "Gaussian process, multi fidelity")
    end = time()
    print("Elapsed time: ", end-start)
    #with open("medium_multi_fidelity_GP.dump" , "wb") as f:
    #    pickle.dump(m_mf, f)  

    # hybrid model, no multifidelity
    print("Gaussian process, single fidelity, delta-learning")
    start = time()
    m_h, k_h = train_gpr(X_medium, y_delta_medium, input_dim)
    metrics_hybrid(m_h, X_test, y_low_as_input_test, y_test, "Gaussian process, single fidelity, delta-learning")
    end = time()
    print("Elapsed time: ", end-start)
    #with open("medium_hybrid_single_fidelity_GP.dump" , "wb") as f:
    #    pickle.dump(m_h, f) 

    # hybrid multi-fidelity model
    print("Gaussian process, multi fidelity, delta-learning")
    start = time()
    model_terra_delta = load_model('C://Users//fedi_vl//Documents//work//terra_deltas_heightmap.h5')
    nn_train_delta = model_terra_delta.predict(x_trans_medium)
    nn_test_delta = model_terra_delta.predict(X_test_trans)
    m_h_mf, k_h_mf = train_mf_gpr(y_low_input, X_medium, y_delta_medium, nn_train_delta, input_dim+1, np.arange(0,input_dim+1))
    metrics_hybrid(m_h_mf, np.hstack((X_test, y_low_as_input_test, nn_test_delta)), y_low_as_input_test, y_test, "Gaussian process, multi fidelity, delta-learning")
    end = time()
    print("Elapsed time: ", end-start)
    #with open("medium_hybrid_multi_fidelity_GP.dump" , "wb") as f:
    #    pickle.dump(m_h_mf, f) 
    '''
    #################### use bayesian NNs as a multi-fidelity method (more like uncertainty propagation method)
    lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
    hu_list = []
    metrics_list = {}
    for x in range(2,5): hu_list += [[y]*x for y in range(5, 10)]
    for learning_rate in lr_list:
        for hidden_units in hu_list:
            current_metrics = []
            ## Single fidelity
            #print("Bayesian NN, single fidelity")
            #start = time()
            bnn_trained = train_mf_bnn(X_medium, y_medium, X_test, y_test, learning_rate, hidden_units)
            m = metrics(bnn_trained, X_test, y_test)
            current_metrics += m
            #end = time()
            #print("Elapsed time: ", end-start)

            ### multi-fidelity
            #print("Bayesian NN, multi fidelity")
            #start = time()
            bnn_trained = train_mf_bnn(np.hstack((np.array(X_medium), y_low_input, nn_train)), y_medium, 
                                    np.hstack((np.array(X_test), y_low[:point_test_medium_fid], nn_test)), y_test,
                                    learning_rate, hidden_units)
            m = metrics(bnn_trained, np.hstack((np.array(X_test), y_low[:point_test_medium_fid], nn_test)), y_test)
            current_metrics += m
            #end = time()
            #print("Elapsed time: ", end-start)
            
            
            ### hybrid single
            #print("Bayesian NN, single fidelity hybrid")
            #start = time()
            bnn_trained = train_mf_bnn(X_medium, y_delta_medium, X_test, y_test, learning_rate, hidden_units)
            m = metrics_hybrid(bnn_trained, X_test, y_low[:point_test_medium_fid], y_test)
            current_metrics += m
            #end = time()
            #print("Elapsed time: ", end-start)
            ## hybrid multi-fidelity
            #print("Bayesian NN, multi-fidelity, hybrid")
            #start = time()
            bnn_trained =train_mf_bnn(np.hstack((np.array(X_medium), y_low_input, nn_train)), y_delta_medium, 
                                    np.hstack((np.array(X_test), y_low[:point_test_medium_fid], nn_test)), y_test,
                                    learning_rate, hidden_units)
            m = metrics_hybrid(bnn_trained, np.hstack((np.array(X_test), y_low[:point_test_medium_fid], nn_test)), y_low[:point_test_medium_fid], y_test)
            current_metrics += m
            #end = time()
            #print("Elapsed time: ", end-start)
            metrics_list['learning_rate ' + str(learning_rate) + " & units : " + str(hidden_units)] = current_metrics
    mdf = pd.DataFrame(metrics_list)
    mdf.to_csv("metrics_gpr.csv", index=False, header=False)'''