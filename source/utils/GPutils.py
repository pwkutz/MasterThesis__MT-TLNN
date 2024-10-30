import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, DotProduct, ConstantKernel as C, Matern, WhiteKernel, RationalQuadratic
import math
import numpy as np
import timeit

#IterGP
from probnum import linops, randvars
from probnum.randprocs import kernels, mean_fns
from itergp import GaussianProcess
from itergp import methods
from probnum import backend

from itergp.methods import policies

from source.calibration_utils import _cggp as cggp

# # GPflow
import gpflow

# GPyTorch
import math
import torch
import gpytorch

# netcal
from netcal import cumulative_moments


# Common interface for all the GP packages I used throughout my project so that I could easily switch them out


#  ██████╗  █████╗ ██╗   ██╗███████╗███████╗██╗ █████╗ ███╗   ██╗    ██████╗ ██████╗  ██████╗  ██████╗███████╗███████╗███████╗    ██████╗ ███████╗ ██████╗ ██████╗ ███████╗███████╗███████╗██╗ ██████╗ ███╗   ██╗
# ██╔════╝ ██╔══██╗██║   ██║██╔════╝██╔════╝██║██╔══██╗████╗  ██║    ██╔══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝██╔════╝    ██╔══██╗██╔════╝██╔════╝ ██╔══██╗██╔════╝██╔════╝██╔════╝██║██╔═══██╗████╗  ██║
# ██║  ███╗███████║██║   ██║███████╗███████╗██║███████║██╔██╗ ██║    ██████╔╝██████╔╝██║   ██║██║     █████╗  ███████╗███████╗    ██████╔╝█████╗  ██║  ███╗██████╔╝█████╗  ███████╗███████╗██║██║   ██║██╔██╗ ██║
# ██║   ██║██╔══██║██║   ██║╚════██║╚════██║██║██╔══██║██║╚██╗██║    ██╔═══╝ ██╔══██╗██║   ██║██║     ██╔══╝  ╚════██║╚════██║    ██╔══██╗██╔══╝  ██║   ██║██╔══██╗██╔══╝  ╚════██║╚════██║██║██║   ██║██║╚██╗██║
# ╚██████╔╝██║  ██║╚██████╔╝███████║███████║██║██║  ██║██║ ╚████║    ██║     ██║  ██║╚██████╔╝╚██████╗███████╗███████║███████║    ██║  ██║███████╗╚██████╔╝██║  ██║███████╗███████║███████║██║╚██████╔╝██║ ╚████║
#  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚══════╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝

# --------------------------> sklearn <--------------------------

# tune the kernel hyperparameters using sklearn kernels
# (done with a seperate GP so that the same kernel can be used for different GP calculations)


def fit_kernel(X_train, Y_train, kernel, hyperparameter_optimizer=None, noise_std=0.0):
    gp = GaussianProcessRegressor(
        kernel=kernel, optimizer=hyperparameter_optimizer, alpha=noise_std**2, n_restarts_optimizer=10
    )
    gp.fit(X_train, Y_train)
    print("fitted kernel: %s" % gp.kernel_)
    return gp.kernel_

# GP regression using sklearn (calculates the exact posterior)


def GP_Regression_sklearn(X_train, Y_train, X_test, kernel, noise_var=0.0):
    gp = GaussianProcessRegressor(
        kernel=kernel, optimizer=None, alpha=noise_var
    )
    gp.fit(X_train, Y_train)
    mean, std = gp.predict(X_test, return_std=True)
    return mean, std**2

# --------------------------> GPflow <--------------------------


def fit_model_legacy(X_train, Y_train, kernel):
    model = gpflow.models.GPR((X_train, Y_train.reshape(-1, 1)), kernel)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    gpflow.utilities.print_summary(model)
    return gpflow.utilities.parameter_dict(model)

# more convenient version with data object


def fit_model(data, kernel):
    model = gpflow.models.GPR(
        (data.X_train, data.Y_train.reshape(-1, 1)), kernel)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    gpflow.utilities.print_summary(model)
    return gpflow.utilities.parameter_dict(model)

# note: the kernel has to be the same as the one used for fitting the parameters

# returns the mean and VARIANCE


def GP_Regression_GPflow_legacy(X_train, Y_train, X_test, kernel, params=None):
    model = gpflow.models.GPR((X_train, Y_train.reshape(-1, 1)), kernel)
    if params is not None:
        gpflow.utilities.multiple_assign(model, params)
    gpflow.utilities.print_summary(model)
    mean, var = model.predict_f(X_test)
    return np.array(mean).reshape(-1), np.array(var).reshape(-1)

# more convenient version with data object and automatically scales back output


def GP_Regression_GPflow(data, kernel, params=None, calibrate=False, noise_variance=None, portionsize=None, calculate_var=True):
    X_test = data.X_test if not calibrate else data.X_calibration 
    model = gpflow.models.GPR(
        (data.X_train, data.Y_train.reshape(-1, 1)), kernel, noise_variance=noise_variance)
    if params is not None:
        gpflow.utilities.multiple_assign(model, params)        

    gpflow.utilities.print_summary(model)

    #inference everything at once
    if portionsize is None:
        print(X_test.shape)
        print(data.n_features)
        mean, var = model.predict_f(X_test)
        return data.scale_back(np.array(mean).reshape(-1), np.array(var).reshape(-1)) if calculate_var else data.scale_back(np.array(mean).reshape(-1), None)

    #make inference one portion at a time
    else:
        if calculate_var:
            mean = []
            var = []
            for i in range(math.ceil(X_test.shape[0]/portionsize)):
                mean_i, var_i = model.predict_f(X_test[i*portionsize:min((i+1)*portionsize, X_test.shape[0])])
                mean.extend(mean_i)
                var.extend(var_i)
            mean = np.array(mean).flatten()
            var = np.array(var).flatten()
            return data.scale_back(mean, var)
        else:
            mean = []
            for i in range(math.ceil(X_test.shape[0]/portionsize)):
                mean_i, _ = model.predict_f(X_test[i*portionsize:min((i+1)*portionsize, X_test.shape[0])])
                mean.extend(mean_i)
            mean = np.array(mean).flatten()
            return data.scale_back(mean, None)



def time_condition_GPflow(data, kernel, params=None):
    model = gpflow.models.GPR(
        (data.X_train, data.Y_train.reshape(-1, 1)), kernel)
    if params is not None:
        gpflow.utilities.multiple_assign(model, params)
    gpflow.utilities.print_summary(model)

    def wrapper():
        model.predict_f(data.X_test)

    # 'number' specifies how many times to call the function
    timer = timeit.Timer("wrapper()", globals=locals())
    execution_time = timer.timeit(number=1)
    print(f"Time taken for GPflow: {execution_time} seconds")


# --------------------------> IterGP <--------------------------
# GP regression using itergp (calculates an approximate posterior)


def GP_Regression_Iter_legacy(X_train, Y_train, X_test, kernel, approx_method=methods.CG, noise_var=0.0, maxIter=4):
    n_train = Y_train.shape[0]
    mean_fn = mean_fns.Zero(input_shape=(X_train.shape[1],), output_shape=())
    gp = GaussianProcess(mean_fn, kernel)
    noise = randvars.Normal(
        mean=backend.zeros(Y_train.shape),
        cov=linops.Scaling(noise_var, shape=(n_train, n_train)),
    )
    itergp_method = approx_method(maxiter=maxIter)

    gp_post = gp.condition_on_data(
        X_train, Y_train, b=noise, approx_method=itergp_method)
    return gp_post.mean(X_test), gp_post.var(X_test)

# more convenient with data object and automatically scales back output


def GP_Regression_Iter(data, kernel, approx_method=methods.CG(maxiter=4), noise_var=0.0, portionsize=None):
    n_train = data.Y_train.shape[0]
    mean_fn = mean_fns.Zero(input_shape=(
        data.X_train.shape[1],), output_shape=())
    gp = GaussianProcess(mean_fn, kernel)
    noise = randvars.Normal(
        mean=backend.zeros(data.Y_train.shape),
        cov=linops.Scaling(noise_var, shape=(n_train, n_train)),
    )
    itergp_method = approx_method

    gp_post = gp.condition_on_data(
        data.X_train, data.Y_train, b=noise, approx_method=itergp_method)
    if portionsize is None:
        return data.scale_back(gp_post.mean(data.X_test), gp_post.var(data.X_test))
    else:
        mean = []
        var = []
        for i in range(math.ceil(data.X_test.shape[0]/portionsize)):
            mean_i, var_i = (gp_post.mean(data.X_test[i*portionsize:min((i+1)*portionsize, data.X_test.shape[0])]), gp_post.var(data.X_test[i*portionsize:min((i+1)*portionsize, data.X_test.shape[0])]))
            mean.extend(mean_i)
            var.extend(var_i)
        mean = np.array(mean).squeeze()
        var = np.array(var).squeeze()
        return data.scale_back(mean, var)


# def time_condition_Iter(X_train, Y_train, X_test, kernel, approx_method=methods.CG, noise_var=0.0, maxIter=4):
#     n_train = Y_train.shape[0]
#     mean_fn = mean_fns.Zero(input_shape=(X_train.shape[1],), output_shape=())
#     gp = GaussianProcess(mean_fn, kernel)
#     noise = randvars.Normal(
#         mean=backend.zeros(Y_train.shape),
#         cov=linops.Scaling(noise_var, shape=(n_train, n_train)),
#     )
#     itergp_method = approx_method(maxiter=maxIter)
#     # 'number' specifies how many times to call the function

#     def wrapper():
#         gp_post = gp.condition_on_data(
#             X_train, Y_train, b=noise, approx_method=itergp_method)
#         gp_post.mean(X_test)
#         gp_post.var(X_test)

#     timer = timeit.Timer("wrapper()", globals=locals())
#     execution_time = timer.timeit(number=1)
#     print(f"Time taken for IterGP: {execution_time} seconds")


def GP_Regression_Iter_RegularCG(data, kernel, noise_var=0.0, maxIter=4, atol=1e-10, rtol=1e-10, portionsize=None):
    n_train = data.Y_train.shape[0]
    mean_fn = mean_fns.Zero(input_shape=(
        data.X_train.shape[1],), output_shape=())
    gp = GaussianProcess(mean_fn, kernel)
    noise = randvars.Normal(
        mean=backend.zeros(data.Y_train.shape),
        cov=linops.Scaling(noise_var, shape=(n_train, n_train)),
    )
    gp_post = cggp.ConjugateGradientGaussianProcess(
        prior=gp, X=data.X_train, Y=data.Y_train, b=noise, maxiter=maxIter, atol=atol, rtol=rtol)
    if portionsize is None:
        return data.scale_back(gp_post.mean(data.X_test), gp_post.var(data.X_test))
    else:
        mean = []
        var = []
        for i in range(math.ceil(data.X_test.shape[0]/portionsize)):
            mean_i, var_i = (gp_post.mean(data.X_test[i*portionsize:min((i+1)*portionsize, data.X_test.shape[0])]), gp_post.var(data.X_test[i*portionsize:min((i+1)*portionsize, data.X_test.shape[0])]))
            mean.extend(mean_i)
            var.extend(var_i)
        mean = np.array(mean).squeeze()
        var = np.array(var).squeeze()
        return data.scale_back(mean, var)


# # --------------------------> GPyTorch <--------------------------
# torch.set_default_dtype(torch.double)


# def GP_regression_GPyTorch(X_train, Y_train, X_test, kernel, noise_var=0.0, maxIter=4):
#     # convert data to torch tensors
#     X_train = torch.from_numpy(X_train).double()
#     Y_train = torch.from_numpy(Y_train).double()
#     X_test = torch.from_numpy(X_test).double()
#     noise_var = torch.tensor(noise_var).double()

#     class CGGPModel(gpytorch.models.ExactGP):
#         def __init__(self, train_x, train_y, likelihood):
#             super(CGGPModel, self).__init__(train_x, train_y, likelihood)
#             self.mean_module = gpytorch.means.ZeroMean()
#             self.covar_module = kernel

#         def forward(self, x):
#             mean_x = self.mean_module(x)
#             covar_x = self.covar_module(x)
#             return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#     # initialize likelihood and model
#     likelihood = gpytorch.likelihoods.GaussianLikelihood()
#     #likelihood.noise = noise_var
#     likelihood.noise_covar.initialize(noise=(noise_var,))
#     model = CGGPModel(X_train,
#                       Y_train, likelihood)

#     # Get into evaluation (predictive posterior) mode
#     model.eval()
#     likelihood.eval()

#     # , gpytorch.settings.max_preconditioner_size(0):
#     # , gpytorch.settings.max_lanczos_quadrature_iterations(0):
#     # , gpytorch.settings.max_cg_iterations(maxIter):
#     # gpytorch.settings.max_preconditioner_size(0)
#     with torch.no_grad(), gpytorch.settings.cg_tolerance(0.0001):
#         f_preds = model(X_test)
#         return f_preds.mean, f_preds.variance

# ██████╗ ██████╗  ██████╗  ██████╗███████╗███████╗███████╗    ██████╗ ███████╗███████╗██╗   ██╗██╗     ████████╗███████╗
# ██╔══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝██╔════╝    ██╔══██╗██╔════╝██╔════╝██║   ██║██║     ╚══██╔══╝██╔════╝
# ██████╔╝██████╔╝██║   ██║██║     █████╗  ███████╗███████╗    ██████╔╝█████╗  ███████╗██║   ██║██║        ██║   ███████╗
# ██╔═══╝ ██╔══██╗██║   ██║██║     ██╔══╝  ╚════██║╚════██║    ██╔══██╗██╔══╝  ╚════██║██║   ██║██║        ██║   ╚════██║
# ██║     ██║  ██║╚██████╔╝╚██████╗███████╗███████║███████║    ██║  ██║███████╗███████║╚██████╔╝███████╗   ██║   ███████║
# ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚══════╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝

# Error measures


def NLL(mean_prediction, std_prediction, Y):
    n = Y.shape[0]
    nll = -1/n * np.sum(np.log(norm.pdf(Y, mean_prediction, std_prediction)))
    return nll


def RMSE(mean_prediction, Y):
    n = Y.shape[0]
    rmse = np.sqrt(1/n * np.sum((mean_prediction - Y)**2))
    return rmse

def MPIW(X, p, kind="meanstd"):
    if kind == "meanstd":
        (mean, std) = X
        y_lower = mean - norm.ppf(1-p/2) * std
        y_upper = mean + norm.ppf(1-p/2) * std
        return 1/len(mean) * np.sum(y_upper - y_lower)


    elif kind == "cumulative":
        (t_values, cdf) = X
        n = t_values.shape[1]
        y_lower = np.array([np.interp(p/2, cdf[:, k, 0], t_values[:, k, 0]) for k in range(n)])
        y_upper = np.array([np.interp(1-p/2, cdf[:, k, 0], t_values[:, k, 0]) for k in range(n)])
        return 1/n * np.sum(y_upper - y_lower)
    else:
        raise ValueError("kind must be 'meanstd' or 'cumulative'")




