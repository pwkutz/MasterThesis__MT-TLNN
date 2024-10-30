import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from source.utils.deepGP_mudafgp import create_model_deep_gp
from mfgp.adaptation_maximizers import ScipyDirectMaximizer
import mfgp.models as models
import GPy
import matplotlib.pyplot as plt
from tqdm import tqdm

sampling_rate_hf = 10 ### hradcoded in dataset_from_troll.py too... means the frequency sampling for the test data...

def save_weights(model, method, model_name):
    # save weights
    with open('models_weights/' + model_name +'_' + method + '.pkl', 'wb') as file:
        pickle.dump(model, file)

def mse_distr(model, num_tests, X_hf_test, y_hf_test, output_index, model_name, min_hf, max_hf):
    # error distribution and it mean calculation
    error = []
    normalized_error = []
    if model_name == "AR1":
        for test_id in tqdm(range(num_tests), desc="test runs"):
            X_hf_test_scaled = ((X_hf_test[test_id]-min_hf)/(max_hf-min_hf))[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]].copy()
            mean, var = model.predict(tf.convert_to_tensor(X_hf_test_scaled), level=-1) # 
            error.append(mean_squared_error(mean, y_hf_test[test_id][:, output_index][:, None])) #
            normalized_error.append(mean_squared_error(mean, y_hf_test[test_id][:, output_index][:, None])/np.linalg.norm(y_hf_test[test_id][:, output_index][:, None]))
    else:
        for test_id in tqdm(range(num_tests), desc="test runs"):
            X_hf_test_scaled = ((X_hf_test[test_id]-min_hf)/(max_hf-min_hf))[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]].copy()
            mean, var = model.predict(X_hf_test_scaled)
            error.append(mean_squared_error(mean, y_hf_test[test_id][:, output_index][:, None])) #
            normalized_error.append(mean_squared_error(mean, y_hf_test[test_id][:, output_index][:, None])/np.linalg.norm(y_hf_test[test_id][:, output_index][:, None]))
    return error, normalized_error

def train_model(model_name, output_index, lf = None, point_share_hf = None, X_lf = None, X_hf = None, y_lf = None, y_hf = None, num_delays = 0, tau = None, num_inducing_low = None, num_inducing_high = None):

    sampling_vec = np.random.choice(range(X_hf.shape[0]), size = X_hf.shape[0], replace=False)
    input_dim = X_hf.shape[1]
    X_train_hf = X_hf[sampling_vec][:int(len(sampling_vec)*point_share_hf)]
    y_train_hf = y_hf[sampling_vec][:, output_index][:, None][:int(len(sampling_vec)*point_share_hf)]

    if model_name == "AR1":
        lower_bound, upper_bound = np.min(np.concatenate([np.min(X_train_hf, axis=0)[None], np.min(X_lf, axis=0)[None]], axis=0), axis=0), np.max(np.concatenate([np.max(X_train_hf, axis=0)[None], np.max(X_lf, axis=0)[None]], axis=0), axis=0)
        f_list = [None, None]
        X_train = [tf.convert_to_tensor(X_lf), tf.convert_to_tensor(X_train_hf)]
        model = models.AR1(input_dim, f_list, lower_bound, upper_bound)
        model.set_training_data(X_train, [tf.convert_to_tensor(y_lf[:, output_index][:, None]), tf.convert_to_tensor(y_train_hf)])
        model.fit()
        print("Negative of likelihood before optimisation", model.neg_unnormalised_log_likelihood())
        model.ARD()
        print("negative of likelihood after optimisation", model.neg_unnormalised_log_likelihood())

    elif model_name in ["NARGP", "GPDF", "GPDFC"]:
        lower_bound, upper_bound = np.min(np.concatenate([np.min(X_train_hf, axis=0)[None], np.min(X_lf, axis=0)[None]], axis=0), axis=0), np.max(np.concatenate([np.max(X_train_hf, axis=0)[None], np.max(X_lf, axis=0)[None]], axis=0), axis=0)
        if model_name == "NARGP":
            model = models.NARGP(input_dim, None, lf, lower_bound, upper_bound)
        else:
            # GPDF(C)
            model = models.__getattribute__(model_name)(input_dim, 1.0, 1, None, lf, lower_bound, upper_bound)
        model.fit_with_val(X_train_hf, y_train_hf)

    elif model_name in ["NARDGP", "DGPDF", "DGPDFC"]:
        lower_bound, upper_bound = np.min(np.concatenate([np.min(X_train_hf, axis=0)[None], np.min(X_lf, axis=0)[None]], axis=0), axis=0), np.max(np.concatenate([np.max(X_train_hf, axis=0)[None], np.max(X_lf, axis=0)[None]], axis=0), axis=0)
    
        _, model = create_model_deep_gp(model_name, input_dim, [lf, None], X_lf, X_train_hf, y_lf[:, output_index][:,None], y_train_hf, num_delays, tau, lower_bound, upper_bound, ScipyDirectMaximizer(), num_inducing_low, num_inducing_high, eps=1e-6)

    elif model_name == "single":
        k1 = GPy.kern.RBF(input_dim, ARD=False)
        model = GPy.models.GPRegression(X=X_train_hf, 
                                        Y=y_train_hf, kernel=k1)
        model.optimize(max_iters = 500)

    return model

def train_zoo(n_all_runs, train_share, X_hf_test, y_hf_test, output_index, lf_gp_sur, sampling_vec, points_share, X_lf, X_hf, y_lf, y_hf, num_delays = 1, tau = 0.1):
    errors_wrt_samplimg = {}

    for point_share_hf in points_share: # arbitrary chosen numbers of HF training points
        errors_scm_as_lf = []
        for model_name in ["single", "AR1", "NARGP", "GPDF", "GPDFC", "NARDGP", "DGPDF", "DGPDFC"]: # list all the methods included here
            if model_name in ["DGPDF", "DGPDFC"]:
                model = train_model(model_name, output_index, lf_gp_sur, sampling_vec, point_share_hf, X_lf, X_hf, y_lf, y_hf, num_delays = num_delays, tau = tau)
                errors_scm_as_lf.append(mse_distr(model, int(n_all_runs*(1-train_share)), X_hf_test, y_hf_test, 0, model_name))
            else:
                model = train_model(model_name, output_index, lf_gp_sur, sampling_vec, point_share_hf, X_lf, X_hf, y_lf, y_hf, num_delays = 0, tau = 0)
                errors_scm_as_lf.append(mse_distr(model, int(n_all_runs*(1-train_share)), X_hf_test, y_hf_test, 0, model_name))
        errors_wrt_samplimg[point_share_hf] = errors_scm_as_lf

    with open('D:/Vlad/mufintroll/results/results_' + lf_gp_sur.__name__ + '.pkl', 'wb') as file:
        pickle.dump(errors_wrt_samplimg, file)

    return errors_wrt_samplimg

def visualize_mf(output_index, lf_gp_sur, sampling_vec, points_share, X_lf, X_hf, y_lf, y_hf, X_hf_test, y_hf_test, train_test_dataset_from_troll):
    nardgp = train_model("NARDGP", output_index, lf_gp_sur, sampling_vec, points_share, X_lf, X_hf, y_lf, y_hf, num_delays = 0, tau = 0)
    single_gp = train_model("single", output_index, lf_gp_sur, sampling_vec, points_share, X_lf, X_hf, y_lf, y_hf, num_delays = 0, tau = 0)
    for i in range(len(X_hf_test)):
        mean, var = nardgp.predict(X_hf_test[i])
        mean_singl, var_singl = single_gp.predict(X_hf_test[i])

        plt.figure(figsize=(15, 5))
        plt.plot(np.linspace(0, len(X_hf_test[i])*sampling_rate_hf/60, len((X_hf_test[i]))), np.array(mean).ravel(), label = "NARDGP prediction") 
        plt.fill_between(np.linspace(0, len(X_hf_test[i])*sampling_rate_hf/60, len(X_hf_test[i])),
                np.array(mean).ravel() - np.sqrt(np.abs(var)).ravel(), 
                np.array(mean).ravel() + np.sqrt(np.abs(var)).ravel(), 
                alpha = 0.2, ec='None', label='NARDGP 95% confidence')
        plt.plot(np.linspace(0, len(X_hf_test[i])*sampling_rate_hf/60, len((X_hf_test[i]))), np.array(mean_singl).ravel(), label = "single fidelity prediction") 
        plt.fill_between(np.linspace(0, len(X_hf_test[i])*sampling_rate_hf/60, len(X_hf_test[i])),
                np.array(mean_singl).ravel() - np.sqrt(np.abs(var_singl)).ravel(), 
                np.array(mean_singl).ravel() + np.sqrt(np.abs(var_singl)).ravel(), 
                alpha = 0.2, ec='None', label='single fidelity 95% confidence')
        plt.plot(np.linspace(0, len(X_hf_test[i])*sampling_rate_hf/60, len(X_hf_test[i])), y_hf_test[i][:, 0][:, None], label = "TROLL")
        plt.plot(np.linspace(0, len(X_hf_test[i])*sampling_rate_hf/60, len(X_hf_test[i])), train_test_dataset_from_troll.dataset['scm'].train_test_dataset['test']['y'][i][:, 0][:, None], label = "SCM")
        plt.xlabel("run time, [s]")
        plt.ylabel("traction force, [N]")
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.savefig('D://Vlad/plots_for_MuDaFuGP/preds_vis/preds_vis' + str(i) + '.pdf', bbox_inches='tight')

def compare_errors(train_test_dataset_from_troll):
    with open('results/results_lf_gp_sur_scm.pkl', 'rb') as f:
        x = pickle.load(f)
    #x = terra_errors_wrt_samplimg
    [np.ma.masked_invalid(x[1][i][1]).mean() for i in range(8)]
    [np.ma.masked_invalid(x[0.5][i][1]).mean() for i in range(8)]
    [np.ma.masked_invalid(x[0.25][i][1]).mean() for i in range(8)]
    [np.ma.masked_invalid(x[0.1][i][1]).mean() for i in range(8)]
    [np.ma.masked_invalid(x[0.05][i][1]).mean() for i in range(8)]

    np.ma.masked_invalid(train_test_dataset_from_troll.return_sim_mse("scm")/np.array([np.linalg.norm(train_test_dataset_from_troll.dataset['troll'].train_test_dataset['test']['y'][x][:, 0]) for x in range(22)])).mean()
    np.ma.masked_invalid(train_test_dataset_from_troll.return_sim_mse("terra")/np.array([np.linalg.norm(train_test_dataset_from_troll.dataset['troll'].train_test_dataset['test']['y'][x][:, 0]) for x in range(22)])).mean()