import netcal as netcal
from netcal.regression import IsotonicRegression, GPBeta, GPNormal
from netcal.metrics import PinballLoss, NLL, ENCE
from netcal import cumulative_moments
from netcal.presentation import ReliabilityRegression
import numpy as np
from source.calibration_utils import visualization
from source.utils.GPutils import MPIW, RMSE
from source.utils.preprocessing import gauss_ma

desired_length = 50
window = 20

class Calibration():
    """
    Calibration is a post-processing of predicted ucnertainties. We implement here three different calibration techniques: \n
    * Isotonic calibration (quantile calibration) \n
    * Beta calibration (variance calibration) \n
    * Normal calibration (variance calibration) \n
    The code was adapted from the thesis of my bachelor student, Jana Huhne. See for theoretical details: https://mediatum.ub.tum.de/doc/1728130/document.pdf
    """
    def __init__(self, model, X_test, y_test, output_indx = 0):
        self.X_calibr = X_test #X_test[:int(len(X_test)/2)]
        self.X_test = X_test #X_test[int(len(X_test)/2):]
        self.y_calibr = y_test #y_test[:int(len(X_test)/2)]    
        self.y_test = y_test #y_test[int(len(X_test)/2):]
        self.y_ground_truth_calibr = y_test[:, output_indx] #np.concatenate(self.y_calibr)[:, output_indx]
        self.y_ground_truth_test = y_test[:, output_indx] #np.concatenate(self.y_test)[:, output_indx]
        self.X_ground_truth_calibr = X_test #np.concatenate(self.X_calibr)
        self.X_ground_truth_test = X_test #np.concatenate(self.X_test)
        self.mean_calib, self.var_calib = model.predict(self.X_ground_truth_calibr)
        self.mean_test, self.var_test = model.predict(self.X_ground_truth_test)

        #self.y_ground_truth_calibr = gauss_ma(self.y_ground_truth_calibr, desired_length, window)
        #self.y_ground_truth_test = gauss_ma(self.y_ground_truth_test, desired_length, window)

        #self.mean_calib, self.var_calib = gauss_ma(np.array(self.mean_calib).ravel(), desired_length, window), gauss_ma(np.array(self.var_calib).ravel(), desired_length, window)
        #self.mean_test, self.var_test = gauss_ma(np.array(self.mean_test).ravel(), desired_length, window), gauss_ma(np.array(self.var_test).ravel(), desired_length, window)

    def isotonic_calibration(self):
        """
        Quantile calibration, analogous to the Platt scalling in classification. It ensures that quantiles of the true distribution are matching with the quantiles of predictions, i.e. distributions is calibrated if predicted X% CI must contain X% of the ground truth points in its CI. And this should hold for all X%. \n
        https://proceedings.mlr.press/v80/kuleshov18a.html
        """
        self.isotonic = IsotonicRegression()
        #fit the isotonic regression
        self.isotonic.fit((self.mean_calib, np.sqrt(self.var_calib)), self.y_ground_truth_calibr)
        #transform the predictions
        t_iso_test, pdf_iso_test, cdf_iso_test = self.isotonic.transform((np.array(self.mean_test).ravel(), np.sqrt(np.array(self.var_test).ravel())))
        return t_iso_test, pdf_iso_test, cdf_iso_test

    def beta_calibration(self):
        """
        Beta map is one of the distribution calibration methods. Its defined on CDF and distributed as Beta-distribution, hence the name.
        """
        self.gpbeta = GPBeta(
            n_inducing_points=64,    # number of inducing points
            n_random_samples=128,    # random samples used for likelihood
            n_epochs=128,            # optimization epochs
            use_cuda=False,          # can also use CUDA for computations
            lr=1e-3,                 # learning rate
        )
        #fit the GP beta model
        self.gpbeta.fit((self.mean_calib, np.sqrt(self.var_calib)), self.y_ground_truth_calibr)
        #transform the predictions
        t_gpbeta_test, pdf_gpbeta_test, cdf_gpbeta_test = self.gpbeta.transform((self.mean_test, np.sqrt(self.var_test)))
        return t_gpbeta_test, pdf_gpbeta_test, cdf_gpbeta_test

    def normal_calibration(self):
        """
        Normal calibration is just a scalling factor of the variance in predicted/posterior GP. Becuase we have only one scaller, it can be distributed as 1D Gaussian process.
        For theoretical details: https://link.springer.com/chapter/10.1007/978-3-031-25072-9_30
        """
        #NOTE (for Variance and Beta): had to modify line 693 in AbstractGP.py from netcal from 
        # with gpytorch.settings.cholesky_jitter(float=self.jitter, double=self.jitter), tqdm(total=self.n_epochs) as pbar:
        # to
        # with gpytorch.settings.cholesky_jitter(float_value=self.jitter, double_value=self.jitter), tqdm(total=self.n_epochs) as pbar:
        self.gpnormal = GPNormal(
            n_inducing_points=64,    # number of inducing points
            n_random_samples=128,    # random samples used for likelihood
            n_epochs=128,            # optimization epochs
            use_cuda=False,          # can also use CUDA for computations
            lr=1e-3,                 # learning rate
        )
        #fit the GP normal model
        self.gpnormal.fit((self.mean_calib, np.sqrt(self.var_calib)), self.y_ground_truth_calibr)
        #transfrom the predictions
        std_gpnormal_test = self.gpnormal.transform((self.mean_test, np.sqrt(self.var_test)))
        return std_gpnormal_test

    def visualize_misscalibration(self, t_iso_test, cdf_iso_test, t_gpbeta_test, cdf_gpbeta_test, std_gpnormal_test):
        # define the quantile levels that are used for the quantile evaluation
        quantiles = np.linspace(0.0, 1.0, 20)
        # initialize the diagram object
        diagram = ReliabilityRegression(quantiles=quantiles)
        # BEFORE CALIBRATION
        diagram.plot((self.mean_test, np.sqrt(self.var_test)), self.y_ground_truth_test, title_suffix="uncalibrated")#, tikz=True, filename="Calibration/Reliability_Uncalibrated.tikz")#, fig=fig)
        # ISOTONIC CALIBRATION
        diagram.plot((t_iso_test, cdf_iso_test), self.y_ground_truth_test, kind='cumulative', title_suffix="Isotonic calibration")#, tikz=True, filename="Calibration/Reliability_Isotonic.tikz")
        # GPBeta CALIBRATION
        diagram.plot((t_gpbeta_test, cdf_gpbeta_test), self.y_ground_truth_test, kind='cumulative', title_suffix="GP Beta calibration")#, tikz=True, filename="Calibration/Reliability_Beta.tikz")
        # GPNormal CALIBRATION
        diagram.plot((self.mean_test, std_gpnormal_test), self.y_ground_truth_test, title_suffix="GP Normal calibration")#, tikz=True, filename="Calibration/Reliability_Normal.tikz")

    def visualization(self, std_gpnormal_test, t_iso_test, cdf_iso_test, t_gpbeta_test, cdf_gpbeta_test, indexes_list):
        #selected range of test data to display (everything would be too large to see anything)
        if len(indexes_list) > 1:
            cum = 0
            for index in indexes_list:
                a = cum
                b = cum + len(self.X_test[int(len(self.X_test)/2):][index])
                #print(a)
                #print(b)
                cum += len(self.X_test[int(len(self.X_test)/2):][index])
                results_normal = [
                    (self.mean_test[a:b], np.sqrt(self.var_test).flatten()[a:b], "Uncalibrated"),
                    (self.mean_test[a:b], std_gpnormal_test.flatten()[a:b], "GP Normal calibration"),
                ]

                results_cdf = [
                    (t_iso_test[:,a:b,:], cdf_iso_test[:,a:b, :], "Isotonic calibration"),
                    (t_gpbeta_test[:,a:b,:], cdf_gpbeta_test[:,a:b, :], "GP Beta calibration"),
                ]
                sampling_rate_hf = 10 ### hradcoded in dataset_from_troll.py too... means the frequency sampling for the test data...
                x_time = np.linspace(0, len(self.y_ground_truth_test[a:b])*sampling_rate_hf/60, len(self.y_ground_truth_test[a:b]))
                visualization.visualize_calibration(x_time, self.y_ground_truth_test[a:b], results_normal, results_cdf, [0.95], index)
                #visualization.visualize_calibration_map_Beta(self.gpbeta, [(-20,2), (-5, 2), (10,2), (-5, 4)])#, tikzpath = "Calibration/BetaMap.tikz", csvpath="Calibration/BetaMap.csv")
                #visualization.visualize_calibration_map_Normal(self.gpnormal, [(-20,2), (-5, 2), (10,2), (-5, 4)])#, tikzpath = "Calibration/NormalMap.tikz", csvpath="Calibration/NormalMap.csv")
                #visualization.visualize_calibration_map_Iso(self.isotonic)#, tikzpath="Calibration/IsoMap.tikz", csvpath="Calibration/IsotonicMap.csv")
        else:
            index = indexes_list[0]
            results_normal = [
                (self.mean_test, np.sqrt(self.var_test).flatten(), "Uncalibrated"),
                (self.mean_test, std_gpnormal_test.flatten(), "GP Normal calibration"),
            ]

            results_cdf = [
                (t_iso_test, cdf_iso_test, "Isotonic calibration"),
                (t_gpbeta_test, cdf_gpbeta_test, "GP Beta calibration"),
            ]
            sampling_rate_hf = 10 ### hradcoded in dataset_from_troll.py too... means the frequency sampling for the test data...
            print(len(self.y_ground_truth_test)*sampling_rate_hf/60)
            print(len(self.y_ground_truth_test))
            x_time = np.linspace(0, len(self.y_ground_truth_test)*sampling_rate_hf/60, len(self.y_ground_truth_test))
            visualization.visualize_calibration(x_time, self.y_ground_truth_test, results_normal, results_cdf, [0.95], index)

    def calibration_metrics(self, t_gpbeta_test, cdf_gpbeta_test, t_iso_test, cdf_iso_test, std_gpnormal_test):
        """
        Detailed description and formulas of metrics please see here: https://mediatum.ub.tum.de/doc/1728130/document.pdf, Section 3.3.5 Measures for calibration
        """
        # Different calibration metrics used to compare the methods
        ### PINBALL LOSS [Quantile] ###
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Pinball Loss <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        quantiles = np.linspace(0.05, 0.95, 19)
        pbl = PinballLoss()
        print("Uncalibrated:", pbl.measure((self.mean_test, np.sqrt(self.var_test)), self.y_ground_truth_test, q=quantiles, kind = "meanstd", reduction="mean"))
        print("Isotonic Regression:", pbl.measure((t_iso_test, cdf_iso_test), self.y_ground_truth_test, q=quantiles, kind = "cumulative", reduction="mean"))
        print("GP Normal:", pbl.measure((self.mean_test, std_gpnormal_test), self.y_ground_truth_test, q=quantiles, kind = "meanstd", reduction="mean"))
        print("GP Beta:", pbl.measure((t_gpbeta_test, cdf_gpbeta_test), self.y_ground_truth_test, q=quantiles, kind = "cumulative", reduction="mean"))
        ### NLL [Distribution] ###
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> NLL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        nll = NLL()
        print("Uncalibrated:", nll.measure((self.mean_test, np.sqrt(self.var_test)), self.y_ground_truth_test, kind = "meanstd", reduction="mean"))
        print("Isotonic Regression:", nll.measure((t_iso_test, cdf_iso_test), self.y_ground_truth_test, kind = "cumulative", reduction="mean"))
        print("GP Normal:", nll.measure((self.mean_test, std_gpnormal_test), self.y_ground_truth_test, kind = "meanstd", reduction="mean"))
        print("GP Beta:", nll.measure((t_gpbeta_test, cdf_gpbeta_test), self.y_ground_truth_test, kind = "cumulative", reduction="mean"))
        ### ENCE [Variance] ###
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ENCE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        ence = ENCE(bins=10)
        print("Uncalibrated:", ence.measure((self.mean_test, np.sqrt(self.var_test)), self.y_ground_truth_test, kind = "meanstd"))
        print("Isotonic Regression:", ence.measure((t_iso_test, cdf_iso_test), self.y_ground_truth_test, kind = "cumulative"))
        print("GP Normal:", ence.measure((self.mean_test, std_gpnormal_test), self.y_ground_truth_test, kind = "meanstd"))
        print("GP Beta:", ence.measure((t_gpbeta_test, cdf_gpbeta_test), self.y_ground_truth_test, kind = "cumulative"))
        ### MPIW [Sharpness] ###
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MPIW <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        p = 0.95
        print("Uncalibrated:", MPIW((self.mean_test, np.sqrt(self.var_test)),p, kind="meanstd"))
        print("Isotonic Regression:", MPIW((t_iso_test, cdf_iso_test), p, kind="cumulative"))
        print("GP Normal:", MPIW((self.mean_test, np.squeeze(std_gpnormal_test)), p, kind="meanstd"))
        print("GP Beta:", MPIW((t_gpbeta_test, cdf_gpbeta_test), p, kind="cumulative"))
        ### RMSE [Accuracy] ###
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> RMSE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("Uncalibrated:", RMSE(self.mean_test, self.y_ground_truth_test))
        print("Isotonic Regression:", RMSE(np.squeeze(cumulative_moments(t_iso_test, cdf_iso_test)[0]), self.y_ground_truth_test))
        print("GP Normal:", RMSE(self.mean_test, self.y_ground_truth_test))
        print("GP Beta:", RMSE(np.squeeze(cumulative_moments(t_gpbeta_test, cdf_gpbeta_test)[0]), self.y_ground_truth_test))