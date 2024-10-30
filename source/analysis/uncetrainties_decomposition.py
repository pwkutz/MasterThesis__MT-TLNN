### Adaptation from Jana's code
### Probably just need use her's, but align the data partitioning format and related stuff

from source.utils.GPutils import *
from source.calibration_utils.visualization import *
#import probnum
from probnum.randprocs.kernels import Matern, ExpQuad, WhiteNoise
import sklearn as sklearn
#import gc
import GPy

torch.set_default_dtype(torch.double)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scale back the GP model predictions according to the scaling applied to the training data before
def scale_back(mean, var, output_scaler):
    mean = output_scaler.inverse_transform(mean.reshape(-1, 1)).flatten()
    var = output_scaler.var_ * var if var is not None else None
    return mean, var

def GP_Regression_Iter_(X_train, Y_train, X_test, kernel, output_scaler, approx_method=methods.CG(maxiter=4), noise_var=0.0, portionsize=None):
    n_train = Y_train.shape[0]
    mean_fn = mean_fns.Zero(input_shape=(X_train.shape[1],), output_shape=())
    gp = GaussianProcess(mean_fn, kernel)
    noise = randvars.Normal(
        mean=backend.zeros(Y_train.shape),
        cov=linops.Scaling(noise_var, shape=(n_train, n_train)),
    )
    itergp_method = approx_method

    gp_post = gp.condition_on_data(X_train, Y_train, b=noise, approx_method=itergp_method)
    if portionsize is None:
        return scale_back(gp_post.mean(X_test), gp_post.var(X_test), output_scaler)
    else:
        mean = []
        var = []
        for i in range(math.ceil(X_test.shape[0]/portionsize)):
            mean_i, var_i = (gp_post.mean(X_test[i*portionsize:min((i+1)*portionsize, X_test.shape[0])]), 
                             gp_post.var(X_test[i*portionsize:min((i+1)*portionsize, X_test.shape[0])]))
            mean.extend(mean_i)
            var.extend(var_i)
        mean = np.array(mean).squeeze()
        var = np.array(var).squeeze()
        return scale_back(mean, var, output_scaler)
    

def run_decomposition(X_hf, y_hf, X_hf_test, y_hf_test, sampling_rate_hf):
    k1 = GPy.kern.RBF(12)
    singl_model = GPy.models.GPRegression(X=X_hf, Y=y_hf[:, 0][:, None], kernel=k1)
    singl_model.optimize(max_iters = 1000)
    for i in range(len(X_hf_test)):
        #trained non ARD Matern kernel on reduced features 
        kernel_var = singl_model.kern.variance[0]
        kernel_lengthscale = singl_model.kern.lengthscale[0]
        gp_var = singl_model.Gaussian_noise.variance[0]
        #GPflow kernel
        kernel_GPflow = gpflow.kernels.SquaredExponential(variance=kernel_var, lengthscales=kernel_lengthscale)

        #run the gaussian processes
        model = gpflow.models.GPR((X_hf, y_hf[:, 0].reshape(-1, 1)), kernel_GPflow, noise_variance=gp_var)
        mean_gpflow, var_gpflow = model.predict_f(X_hf_test[i])    

        n_features = 12
        num_iterations = 1000
        #IterGP Kernel

        kern1 = kernel_var * ExpQuad(input_shape=(n_features,), lengthscale=kernel_lengthscale)

        kernel_Iter = kern1 
        output_scaler = StandardScaler()
        Y_train = output_scaler.fit_transform(y_hf[:, 0].reshape(-1, 1)).flatten()

        mean_Iter, var_Iter = GP_Regression_Iter_(X_hf, y_hf[:, 0], X_hf_test[i], kernel_Iter, output_scaler, noise_var=1, approx_method=methods.Cholesky(maxrank=num_iterations, atol=1e-10, rtol=1e-10)) 
        # visualize range [a, b) of the test data fter n iterations of IterGP along with the split in computational and mathematical uncertainty
        #a = 20
        #b = 120
        x_time = np.linspace(0, len(X_hf_test[i])*sampling_rate_hf/60, len(X_hf_test[i]))
        visualize_math_iter(x_time, y_hf_test[i][:, 0], np.array(mean_gpflow).flatten(), np.array(var_gpflow).flatten(), mean_Iter, var_Iter)