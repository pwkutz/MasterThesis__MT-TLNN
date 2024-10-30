# own stuff
from source.utils.GPutils import *
#from preprocessing import *
from source.calibration_utils.visualization import *

# SAlib used for sensitivity analysis
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol
from SALib.util import read_param_file, ResultDict, extract_group_names, _check_groups, compute_groups_matrix, scale_samples

# other stuff
import scipy as sp
import pickle
from scipy.stats import qmc
from typing import Dict, Optional, Union
import warnings 
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from source.utils.gp_export import nargp_pred

class sensitivity_analyser():
    def __init__(self, X_lf, y_lf, X_train, y_hf, nardgp):#, scaled_y):
        self.X_lf = X_lf
        self.y_lf = y_lf
        #self.X_hf = X_hf
        self.X_train = X_train
        self.y_hf = y_hf
        self.num_x = 200
        self.nardgp = nardgp
        #self.scaled_y = scaled_y
        #self.surrogate_pred = (surrogate_pred/np.max(surrogate_pred)).copy()
        #self.X_train = np.concatenate([self.X_hf, self.surrogate_pred], axis = 1)
        self.Si = None
        self.Si_distr = []
        self.Si_distr_corr = []

        self.n_features = self.X_train.shape[1]
        self.feature_descriptions = ["p0", "p2", "v0", "v1", "v2", "w0", "w1", "w2", "n0", "n2"]#, "lf"]
        self.maineffects = np.zeros((self.n_features, self.num_x))
        print("maineffects shape: ", self.maineffects.shape)

######################## Jana's code for SA ##################################
# Code used for the sensitivity analysis, I mostly used SAlib but had to augment it with my own for sampling from non-uniform distributions and for working with correlated samples
# generate N independent samples from the sobol sequence and transform them according to the distributions. N should be a power of 2
# Adapted from SAlib
    def generate_samples(self,
        problem: Dict,
        N: int,
        ppf_functions,
        *,
        calc_second_order: bool = True,
        scramble: bool = True,
        skip_values: int = 0,
        seed: Optional[Union[int, np.random.Generator]] = None,
        correlations = None,
        R = None,):

        #STEP 1: sample from the sobol sequence (This step is taken from SAlib)
        D = problem["num_vars"]
        groups = _check_groups(problem)

        # Create base sequence - could be any type of sampling
        qrng = qmc.Sobol(d=2 * D, scramble=scramble, seed=seed)

        # fast-forward logic
        if skip_values > 0 and isinstance(skip_values, int):
            M = skip_values
            if not ((M & (M - 1) == 0) and (M != 0 and M - 1 != 0)):
                msg = f"""
                Convergence properties of the Sobol' sequence is only valid if
                `skip_values` ({M}) is a power of 2.
                """
                warnings.warn(msg, stacklevel=2)

            # warning when N > skip_values
            # see https://github.com/scipy/scipy/pull/10844#issuecomment-673029539
            n_exp = int(np.log2(N))
            m_exp = int(np.log2(M))
            if n_exp > m_exp:
                msg = (
                    "Convergence may not be valid as the number of "
                    "requested samples is"
                    f" > `skip_values` ({N} > {M})."
                )
                warnings.warn(msg, stacklevel=2)

            qrng.fast_forward(M)
        elif skip_values < 0 or not isinstance(skip_values, int):
            raise ValueError("`skip_values` must be a positive integer.")

        # sample Sobol' sequence
        base_sequence = qrng.random(N)

        #STEP 2: transform the samples according to the distributions (by me)
        for i in range(2*D):
            base_sequence[:, i] = ppf_functions[i % D](base_sequence[:, i])

        #STEP 3: optionally impose covariance structure (by me)
        if correlations is not None:
            print("base sequence shape: ", base_sequence.shape)
            # we have to split the base sequence because it is actually 2 samples of the same 9 variables instead of one 18-D sample
            base_sequence[:, range(D)] = self.correlate_data(base_sequence[:, range(D)], correlations)
            base_sequence[:, range(D, 2*D)] = self.correlate_data(base_sequence[:, range(D, 2*D)], correlations)
        
        return base_sequence

    # generate samples with LHS (as opposed to Sobol' sequences) for the crrelated samples
    def generate_samples_LHS(self, problem: Dict, N, ppf_functions=None, correlations = None, R = None):
        D = problem["num_vars"]

        #generate a LHS sample of size N
        qrng = qmc.LatinHypercube(d=D)
        base_sample = qrng.random(N)

        #transform the samples according to the distributions
        if ppf_functions is not None:
            for i in range(D):
                base_sample[:, i] = ppf_functions[i](base_sample[:, i])

        # impose covariance structure (for correlated analysis) OR generate two independent uniform samples (for independent analysis)
        if correlations is not None:
            sample_1 = base_sample
            sample_2 = np.copy(base_sample) #copy bc I am dumb and my method works in place but also returns the result
            sample_1 = self.correlate_data(sample_1, correlations, R=None)
            sample_2 = self.correlate_data(sample_2, correlations, R=None) #second sample with different, randomly generated R
            return np.concatenate((sample_1, sample_2), axis=1)
        else:
            sample_1 = base_sample
            sample_2 = np.copy(base_sample) # second sample is shuffled randomly to get an independent sample
            for i in range(sample_2.shape[1]):
                np.random.shuffle(sample_2[:,i])
            return np.concatenate((sample_1, sample_2), axis=1)


        
    # create sampling matrices A and B as well as their combinations for the optimized calculation of sobol indices in the independent case
    # code adapted from SALib
    def arange_samples(self, problem, N, base_sequence, calc_second_order = True):
        D = problem["num_vars"]
        groups = _check_groups(problem)
        if not groups:
            Dg = problem["num_vars"]
        else:
            G, group_names = compute_groups_matrix(groups)
            Dg = len(set(group_names))

        if calc_second_order:
            saltelli_sequence = np.zeros([(2 * Dg + 2) * N, D])
        else:
            saltelli_sequence = np.zeros([(Dg + 2) * N, D])

        index = 0

        for i in range(N):
            # Copy matrix "A"
            for j in range(D):
                saltelli_sequence[index, j] = base_sequence[i, j]

            index += 1

            # Cross-sample elements of "B" into "A"
            for k in range(Dg):
                for j in range(D):
                    if (not groups and j == k) or (groups and group_names[k] == groups[j]):
                        saltelli_sequence[index, j] = base_sequence[i, j + D]
                    else:
                        saltelli_sequence[index, j] = base_sequence[i, j]

                index += 1

            # Cross-sample elements of "A" into "B"
            # Only needed if you're doing second-order indices (true by default)
            if calc_second_order:
                for k in range(Dg):
                    for j in range(D):
                        if (not groups and j == k) or (
                            groups and group_names[k] == groups[j]
                        ):
                            saltelli_sequence[index, j] = base_sequence[i, j]
                        else:
                            saltelli_sequence[index, j] = base_sequence[i, j + D]

                    index += 1

            # Copy matrix "B"
            for j in range(D):
                saltelli_sequence[index, j] = base_sequence[i, j + D]

            index += 1

        saltelli_sequence = scale_samples(saltelli_sequence, problem)
        return saltelli_sequence

    # helper -  generate a matrix containing random permutations of a set a
    def generate_R(self, N, K):
        a = [sp.stats.norm.ppf((i)/(N+1)) for i in range(N)]
        permutations = set()
        while len(permutations) < K:
            permutations.add(tuple(np.random.permutation(a))) 
        print("shape of R: ", np.array(list(permutations)).T.shape)
        return np.array(list(permutations)).T

            
    # introduce correlations into a set of independent observations X, where C is the desired spearman rank correlation matrix
    def correlate_data(self, X, C, R=None):
        if R is None:
            print("generating new R")
            R = self.generate_R(X.shape[0], X.shape[1])
        elif isinstance(R, str):
            with open(R, 'rb') as f:
                R = pickle.load(f)
        # step 1: corelate the rank matrix (with variance correction)
        print("shape of R: ", R.shape)
        print("ndim of R: ", R.ndim)
        Pt = np.linalg.cholesky(C).T
        T = sp.stats.spearmanr(R, axis=0)[0]
        Qt = np.linalg.cholesky(T).T
        Qt_inv = np.linalg.inv(Qt)
        #debug
        print("shape of Pt: ", Pt.shape)
        print("shape of Qt: ", Qt.shape)
        R_star = R @ Pt @ Qt_inv

        R_star[np.isnan(R_star)] = 0 ## dirty quick-fix. TODO: think why n\a happens here

        #step  2: reorder each column of X according to R_star
        for k in range(X.shape[1]):
            ranks = sp.stats.rankdata(R_star[:, k], method = 'ordinal')  - 1
            X[:, k] = np.sort(X[:, k])[ranks]

        # some diagnostics on how well the correlation worked
        Diff = np.abs(sp.stats.spearmanr(X, axis=0)[0] - C)
        print("~~~ Correlation Results ~~~")
        print("max difference between desired and actual correlation: ", np.max(Diff))
        print("mean difference between desired and actual correlation: ", np.mean(Diff))
        print("std deviation of difference between desired and actual correlation: ", np.std(Diff))

        return X

    # calculate R_star separately for testing
    def calculate_Rstar(self, X, C, R=None):
        if R is None:
            print("generating new R")
            R = self.generate_R(X.shape[0], X.shape[1])
        elif isinstance(R, str):
            with open(R, 'rb') as f:
                R = pickle.load(f)
        # step 1: corelate the rank matrix (with variance correction)
        print("shape of R: ", R.shape)
        print("ndim of R: ", R.ndim)
        Pt = np.linalg.cholesky(C).T
        T = sp.stats.spearmanr(R, axis=0)[0]
        Qt = np.linalg.cholesky(T).T
        Qt_inv = np.linalg.inv(Qt)
        #debug
        print("shape of Pt: ", Pt.shape)
        print("shape of Qt: ", Qt.shape)
        R_star = R @ Pt @ Qt_inv
        return R_star

    #correlate data with given R_star for testing
    def correlate_data_Rstar(self, X, R_star):
        for k in range(X.shape[1]):
            ranks = sp.stats.rankdata(R_star[:, k], method = 'ordinal')  - 1
            X[:, k] = np.sort(X[:, k])[ranks]
        return X


    # calculate the main effect for a single variable
    def main_effects(self, i, N, feature_bounds, num_x, ppf_functions):
        xi_values = np.linspace(feature_bounds[i][0], feature_bounds[i][1], num_x)
        qrng = qmc.Sobol(d=len(ppf_functions), scramble=True)
        base_sample = qrng.random(N)
        
        # transform the samples according to the distributions
        for j in range(len(ppf_functions)):
            base_sample[:, j] = ppf_functions[i](base_sample[:, j])

        # calculate the main effects
        main_effects = []
        Y_values = []
        for xi in tqdm(xi_values, desc="samples"):
            base_sample[:, i] = xi # shuffle the chosen feature
            Y = self.nardgp.predict(base_sample)[0]#.numpy()[:, 0] #nargp_pred(self.X_train, base_sample, self.nargp, self.scaled_y).numpy()[:, 0]
            #Y, _ = GP_Regression_GPflow_legacy(data.X_train, data.Y_train, base_sample, gpflow.kernels.RationalQuadratic(lengthscales = [1.0]*data.n_features), params_fitted)
            main_effects.append(np.mean(Y))
            
        return main_effects

    # compute the main effects for each input feature over its entire range (200 samples)
    def calculate_main_effects(self):
        feature_bounds = []
        for k in range(self.n_features):
            feature_bounds.append([self.X_train[:,k].min(), self.X_train[:,k].max()])

        ppf_functions = []
        for i in range(self.n_features):
            n = len(self.X_train)
            percentiles = np.linspace(0, 100, n)
            grid = np.percentile(self.X_train[:, i], percentiles)

            ppf_data = sp.interpolate.CubicSpline(np.linspace(0, 1, n), grid)
            ppf_functions.append(ppf_data)

        for i in tqdm(range(self.n_features), desc="features"):
            self.maineffects[i] = self.main_effects(i, N=2**14, feature_bounds=feature_bounds, num_x=self.num_x, ppf_functions=ppf_functions)
            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot scatterplot for off-diagonal subplots
            ax.scatter(np.linspace(feature_bounds[i][0], feature_bounds[i][1], self.num_x), self.maineffects[i], alpha=0.5)
            ax.set_xlabel(self.feature_descriptions[i])
            ax.set_ylabel("Main effect")

            # Set title for each subplot
            ax.set_title("Main effect of " + self.feature_descriptions[i])
            plt.show()

        with open('SensitivityAnalysis_main_effects_200samp_2-12.pkl', 'wb') as f:
            pickle.dump(self.maineffects, f)

        # store main effects to csv for plotting in the thesis
        # Create a DataFrame
        df = pd.DataFrame(self.maineffects.T, columns=['y'+str(i+1) for i in range(self.n_features)])
        df.insert(0, 'x', np.linspace(0,1,200))

        # Save the DataFrame to a CSV file
        df.to_csv('maineffects2.csv', index=False)

    def plot_main_effects(self):
        # plot all main effects here, over the normaliozed range [0,1] to make the different variables comparable
        fig, ax = plt.subplots(figsize=(8, 8))

        for i in range(self.n_features):
            #maineffects[i] = main_effects(i, N=2**12, feature_bounds=feature_bounds, num_x=num_x, ppf_functions=ppf_functions)

            # Plot scatterplot for off-diagonal subplots
            ax.scatter(np.linspace(0, 1, self.num_x), self.maineffects[i], alpha=0.5, label = self.feature_descriptions[i])
        ax.legend()
        ax.set_ylabel("Main effect")

            # Set title for each subplot
        ax.set_title("Main effects")
        plt.show()

    def uniform_sampling(self):
        #cgenerate N(2k + 2) samples, N should be a power of 2
        N = 2**14 

        # determine ranges for the features 
        feature_bounds = []
        for k in range(self.n_features):
            feature_bounds.append([self.X_train[:,k].min(), self.X_train[:,k].max()])
        print(feature_bounds)
        # specify the problem for SAlib
        problem = {
            'num_vars': self.n_features,
            'names': self.feature_descriptions,
            'bounds': feature_bounds
        }

        # Generate random samples (uniform over the specified bounds) with Sobol' sequences
        X_sample = sobol_sampler.sample(problem, N)

        # evaluate the model for the sampled parameters
        #Y, _ = GP_Regression_GPflow_legacy(data.X_train, data.Y_train, X_sample, gpflow.kernels.RationalQuadratic(lengthscales = [1.0]*data.n_features), params_fitted)
        Y = self.nardgp.predict(X_sample)[0].numpy()[:, 0] #nargp_pred(self.X_train, X_sample, self.nargp, self.scaled_y).numpy()[:, 0]

        print(problem)
        # Analysis (note: the conf values are half the width of a 95% confidence interval)
        self.Si = sobol.analyze(problem, Y, print_to_console=True)
        #return Si
    
    def transformed_distribution(self, si_as_rv = False):
        #generate N(2k + 2) samples, N should be a power of 2
        N = 2**14

        #data = magically_get_data(range(0, 67), range(432, 499), 1,0,dataset_small=False, dataset_large=True, gravity_xangle=True, slippage=False,standardize_X=True, standardize_Y=True)

        # features are between 0 and 1 bc we transform them afterwards
        feature_bounds = [[0, 1] for i in range(self.n_features)]

        # specify the problem
        problem = {
            'num_vars': self.n_features,
            'names': self.feature_descriptions,
            'bounds': feature_bounds
        }

        # generate percentile functions from the data without assuming any probability distribution (maybe change to other distributions later)
        ppf_functions = []
        for i in range(self.n_features):
            n = len(self.X_train)
            percentiles = np.linspace(0, 100, n)
            grid = np.percentile(self.X_train[:, i], percentiles)

            ppf_data = sp.interpolate.CubicSpline(np.linspace(0, 1, n), grid)
            ppf_functions.append(ppf_data)

        base_sample = self.generate_samples_LHS(problem, N, ppf_functions, correlations=None)
        X_sample = self.arange_samples(problem, N, base_sample)
        self.X_test = X_sample # yeah thats ugly but I don't want t change the interface rn
        #Y, _ = GP_Regression_GPflow(data, gpflow.kernels.Matern32(lengthscales = [1.0]*data.n_features), params_fitted, False, portionsize=1000, calculate_var=False) 
        #print(self.nardgp.predict(self.X_test)[0].numpy().shape)
       
        #print(problem)
        if si_as_rv:
            mean, full_cov = self.nardgp.predict(self.X_test, full_cov=True) #nargp_pred(self.X_train, self.X_test, self.nargp, self.scaled_y).numpy()[:, 0]
            covar = full_cov.numpy().reshape(full_cov.shape[1], full_cov.shape[2])
            mean = mean.numpy()[:, 0]
            #GP_Regression_GPflow(data, gpflow.kernels.Matern32(lengthscales = [1.0]*data.n_features), params_fitted, portionsize=1000, calculate_var=False)
            # Analysis (note: the conf values are half the width of a 95% confidence interval)
            for i in tqdm(range(50), desc = "samples for GPs posterior"): # idk what to put here
                sampl_from_predictive_distr = np.random.multivariate_normal(mean, covar)
                self.Si_distr.append(sobol.analyze(problem, sampl_from_predictive_distr, print_to_console=True, num_resamples=2000))           
        else:
            mean, _ = self.nardgp.predict(self.X_test, full_cov=False) #nargp_pred(self.X_train, self.X_test, self.nargp, self.scaled_y).numpy()[:, 0]
            mean = mean.numpy()[:, 0] #<- if we use NARDGP
            #mean = mean[:, 0] # <- if we use SCM surrogate
            self.Si = sobol.analyze(problem, mean, print_to_console=True, num_resamples=2000)


    def correlated_sampling(self, problem, si_as_rv = False):
        #generate N(2k + 2) samples, N should be a power of 2 (we can use more here because k is the number of groups, not the number of features)
        N = 2**14

        #data = magically_get_data(range(0, 67), range(432, 499), 1,0,dataset_small=False, dataset_large=True, gravity_xangle=True, slippage=False,standardize_X=True, standardize_Y=True)
        #with open('final_trained_params_collection/allfeatures.pkl', 'rb') as f:
        #    params_fitted = pickle.load(f)

        # determine ranges for the features (TODO ask Vlad !!)
        #feature_bounds = [[0, 1] for i in range(self.n_features)]

        # specify the problem
        #problem = {
        #    'num_vars': self.n_features,
        #    'groups': ['p1 + a2', 'v0 + a1 + p2', 'v0 + a1 + p2', 'v1 + a0', 'v2', 'v1 + a0', 'v0 + a1 + p2', 'p1 + a2', 'g'],
        #    'names': self.feature_descriptions,
        #    'bounds': feature_bounds
        #}

        #generate percentile functions from the data without assuming any probability distribution (maybe change to other distributions later)
        ppf_functions = []
        for i in range(self.n_features):
            n = len(self.X_test)
            percentiles = np.linspace(0, 100, n)
            grid = np.percentile(self.X_train[:, i], percentiles)

            ppf_data = sp.interpolate.CubicSpline(np.linspace(0, 1, n), grid)
            ppf_functions.append(ppf_data)

        # create a rank correlation matrix from the data, with a threshold of abs value of > 0.3
        corr, p = sp.stats.spearmanr(self.X_train.T, axis=1)
        alpha = 0.05
        # filter out insignificant values and values below the threshold
        corr[np.abs(corr) < 0.3] = 0
        corr[p > alpha] = 0
        #base_sample = generate_samples(problem, N, ppf_functions, correlations=corr)
        base_sample = self.generate_samples_LHS(problem, N, ppf_functions, correlations=corr)
        #print(base_sample.shape)
        X_sample = self.arange_samples(problem, N, base_sample, calc_second_order = True) #don't have to change anything here bc groups are independent between each other
        #Y = self.nardgp.predict(X_sample)[0].numpy()[:, 0] #nargp_pred(self.X_train, X_sample, self.nargp, self.scaled_y).numpy()[:, 0]
        #Y, _ = GP_Regression_GPflow(data, gpflow.kernels.RationalQuadratic(lengthscales = [1.0]*data.n_features), params_fitted, False) #GP_Regression_GPflow(data.X_train, data.Y_train, X_sample, gpflow.kernels.RationalQuadratic(lengthscales = [1.0]*data.n_features), params_fitted)
        #GP_Regression_GPflow(data, gpflow.kernels.Matern32(lengthscales = [1.0]*data.n_features), params_fitted, portionsize=1000, calculate_var=False)
        
        print(problem)
        if si_as_rv:
            mean, full_cov = self.nardgp.predict(X_sample, full_cov=True) 
            covar = full_cov.numpy().reshape(full_cov.shape[1], full_cov.shape[2])
            mean = mean.numpy()[:, 0]
            # Analysis (note: the conf values are half the width of a 95% confidence interval)
            for i in range(100): # idk what to put here
                sampl_from_predictive_distr = np.random.multivariate_normal(mean, covar)
                self.Si_distr_corr.append(sobol.analyze(problem, sampl_from_predictive_distr, print_to_console=True, num_resamples=2000))
        else:
            mean, _ = self.nardgp.predict(X_sample, full_cov=False) #nargp_pred(self.X_train, self.X_test, self.nargp, self.scaled_y).numpy()[:, 0]
            mean = mean.numpy()[:, 0] #<- if we use NARDGP
            #mean = mean[:, 0] # <- if we use SCM surrogate
            self.Si = sobol.analyze(problem, mean, print_to_console=True, num_resamples=2000)

        # Analysis (note: the conf values are half the width of a 95% confidence interval)

        ## TODO: instead of passing mean Y, pass several times (how many?) samples, drawn from the NARDGP (use np.multivariate with mean and variance?)

        #self.Si = sobol.analyze(problem, Y, print_to_console=True, num_resamples=2000)

        #for visualization:
        self.feature_descriptions = problem['groups']


    def visualization(self, si_as_rv = False):

        # filter out entries where the confidence interval contains 
        S1_filtered = [(s,c,d) for (s,c,d) in zip(self.Si['S1'], self.Si['S1_conf'], self.feature_descriptions) if math.copysign(1,s - c) == math.copysign(1,s + c)]
        ST_filtered = [(s,c,d) for (s,c,d) in zip(self.Si['ST'], self.Si['ST_conf'], self.feature_descriptions) if math.copysign(1,s - c) == math.copysign(1,s + c)]

        S2_filtered = np.copy(self.Si['S2'])
        for i in range(len(self.Si['S2'])):
            for j in range(i + 1, len(self.Si['S2'])):
                if math.copysign(1,self.Si['S2'][i,j] - self.Si['S2_conf'][i,j]) != math.copysign(1,self.Si['S2'][i,j] + self.Si['S2_conf'][i,j]):
                    S2_filtered[i,j] = np.nan


        print("S1_filtered: ", S1_filtered)
        print("ST_filtered: ", ST_filtered)
        print("S2_filtered: ", S2_filtered)
        len(ST_filtered)
        len(S2_filtered)
        np.sum([t[0] for t in S1_filtered])

        # visualize the first order and total indices
        visualize_feature_selection_color_2(self.Si["S1"],self.Si["S1_conf"], True, self.feature_descriptions, color_description="95% confidence interval half width", title = "S1")
        plt.figure()
        visualize_feature_selection_color_2(self.Si["ST"],self.Si["ST_conf"], True, self.feature_descriptions, color_description="95% confidence interval half width", title = "ST")
        plt.figure()
        visualize_feature_selection([t[0] for t in S1_filtered], True, [t[2] for t in S1_filtered], title = "S1 significant effects")
        plt.figure()
        visualize_feature_selection([t[0] for t in ST_filtered], True, [t[2] for t in ST_filtered], title = "ST significant effects")

        # visualize second order effects
        visualize_matrix(self.Si["S2"], self.feature_descriptions, title="all second order effects")
        plt.figure()
        visualize_matrix(S2_filtered.T, self.feature_descriptions, title="Second order significant effects")

        # just curious: higher order effects not explained by the first or second order effects
        #NOTE repeat after having filtered out non significant ones
        def fill_lower_triangle(matrix):
            n = matrix.shape[0]
            upper_triangle = np.triu(matrix, k=1)  # Get the upper triangle values
            filled_matrix = upper_triangle + upper_triangle.T  # Create a symmetric matrix
            np.fill_diagonal(filled_matrix, 0)  # Set diagonal values to 0
            return filled_matrix

        S_2_full = fill_lower_triangle(self.Si["S2"])
        #visualize_matrix(S_2_full, data.feature_descriptions)

        S_test = [self.Si["ST"][i] - self.Si["S1"][i] - sum(S_2_full[i]) for i in range(len(self.Si["ST"]))]
        visualize_feature_selection(S_test, True, self.feature_descriptions)

