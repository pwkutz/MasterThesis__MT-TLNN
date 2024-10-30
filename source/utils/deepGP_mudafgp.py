from mfgp.models import GPDFC_General, GPDF_General, NARGP_General, AR1, SVGPLayer, NARDGPLayer, DGPDFLayer, DGPDFCLayer
import tensorflow as tf
import gpflow
from gpflow.config import default_float
import numpy as np

def create_model_deep_gp(method_name, input_dim, f_list, X_train_low, X_train_high, Y_train_low, Y_train_high,
                         num_delays, tau, lower_bound, upper_bound, maximiser, num_inducing_low, num_inducing_high,
                         eps=1e-6, expected_acq_fn: bool=False):
    X_train_high_tensor = tf.convert_to_tensor(X_train_high, dtype=default_float())
    X_train_low_tensor = tf.convert_to_tensor(X_train_low, dtype=default_float())   
    Y_train_high_tensor = tf.convert_to_tensor(Y_train_high, dtype=default_float())
    Y_train_low_tensor = tf.convert_to_tensor(Y_train_low, dtype=default_float())
    #num_low, num_high = len(Y_train_low), len(Y_train_high)
    #num_inducing_low, num_inducing_high = int(num_low/10), int(num_high)
    Z_low = tf.linspace(lower_bound, upper_bound, num_inducing_low)#[:, None]
    kernel_low = gpflow.kernels.SquaredExponential()
    likelihood_low = gpflow.likelihoods.Gaussian(variance=1e-4)
    likelihood_high = gpflow.likelihoods.Gaussian(variance=1e-4)
    first_layer = SVGPLayer(f_list[0], input_dim, kernel_low, likelihood_low, Z_low, lower_bound, upper_bound, whiten=False)
    first_layer.set_data(X_train_low_tensor, Y_train_low_tensor)
    first_layer.train()
    first_layer.set_all_training_status_variables(False)
    second_layer = None 
    lower_bound_lf_y = [np.min(Y_train_low) - np.std(Y_train_low)]
    upper_bound_lf_y = [np.max(Y_train_low) + np.std(Y_train_low)]

    ### TODO: generalize boundries "expansion": np.concatenate([upper_bound, upper_bound x num_delays, upper_bound_lf_y])

    if method_name == "NARDGP":
        second_layer = NARDGPLayer(f_list[1], input_dim, likelihood_high, num_inducing_high, np.concatenate([lower_bound, lower_bound_lf_y]), 
                                   np.concatenate([upper_bound, upper_bound_lf_y]), first_layer)
    elif method_name == "DGPDF":
        lower_bound = np.array(list(lower_bound) * 2 * num_delays + list(lower_bound))
        upper_bound = np.array(list(upper_bound) * 2 * num_delays + list(upper_bound))
        second_layer = DGPDFLayer(f_list[1], input_dim, likelihood_high, num_inducing_high, np.concatenate([lower_bound, lower_bound_lf_y]), 
                                   np.concatenate([upper_bound, upper_bound_lf_y]), first_layer, num_delays=num_delays, tau=tau)
    elif method_name == "DGPDFC":
        lower_bound = np.array(list(lower_bound) * 2 * num_delays + list(lower_bound))
        upper_bound = np.array(list(upper_bound) * 2 * num_delays + list(upper_bound))
        second_layer = DGPDFCLayer(f_list[1], input_dim, likelihood_high, num_inducing_high, np.concatenate([lower_bound, lower_bound_lf_y]), 
                                   np.concatenate([upper_bound, upper_bound_lf_y]), first_layer, num_delays=num_delays, tau=tau)
    else:
        raise ValueError("Wrong method name")
    gpflow.utilities.set_trainable(likelihood_high.variance, False)
    second_layer.set_data(X_train_high_tensor, Y_train_high_tensor)
    second_layer.train()
    # second_layer.adapt(4)
    return first_layer, second_layer

def create_three_level_model_deep_gp(method_name, input_dim, second_layer, Y_train_low, X_train_high, Y_train_high,
                         num_delays, tau, lower_bound, upper_bound, num_inducing_high):
    X_train_high_tensor = tf.convert_to_tensor(X_train_high, dtype=default_float())
    Y_train_high_tensor = tf.convert_to_tensor(Y_train_high, dtype=default_float())
    # boundries for the target value from the second level
    lower_bound_lf_y = [np.min(Y_train_low) - np.std(Y_train_low)]
    upper_bound_lf_y = [np.max(Y_train_low) + np.std(Y_train_low)]
    likelihood_high = gpflow.likelihoods.Gaussian(variance=1e-4)
    second_layer.set_all_training_status_variables(False)

    if method_name == "NARDGP":
        third_layer = NARDGPLayer(None, input_dim, likelihood_high, num_inducing_high, np.concatenate([lower_bound, lower_bound_lf_y]), 
                                   np.concatenate([upper_bound, upper_bound_lf_y]), second_layer)
    elif method_name == "DGPDF":
        lower_bound = np.array(list(lower_bound) * 2 * num_delays + list(lower_bound))
        upper_bound = np.array(list(upper_bound) * 2 * num_delays + list(upper_bound))
        third_layer = DGPDFLayer(None, input_dim, likelihood_high, num_inducing_high, np.concatenate([lower_bound, lower_bound_lf_y]), 
                                   np.concatenate([upper_bound, upper_bound_lf_y]), second_layer, num_delays=num_delays, tau=tau)
    elif method_name == "DGPDFC":
        lower_bound = np.array(list(lower_bound) * 2 * num_delays + list(lower_bound))
        upper_bound = np.array(list(upper_bound) * 2 * num_delays + list(upper_bound))
        third_layer = DGPDFCLayer(None, input_dim, likelihood_high, num_inducing_high, np.concatenate([lower_bound, lower_bound_lf_y]), 
                                   np.concatenate([upper_bound, upper_bound_lf_y]), second_layer, num_delays=num_delays, tau=tau)
    else:
        raise ValueError("Wrong method name")
    gpflow.utilities.set_trainable(likelihood_high.variance, False)
    third_layer.set_data(X_train_high_tensor, Y_train_high_tensor)
    third_layer.train()
    return third_layer