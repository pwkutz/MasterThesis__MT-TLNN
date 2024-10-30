import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import numpy as np
import tikzplotlib
# netcal
from netcal import cumulative_moments

from matplotlib import rc
import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)
#mpl.rcParams['text.usetex'] = True

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
# plt.style.use('ggplot')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20

np.random.seed(10)

rc('font', size=BIGGER_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# visualize results, x has to be one dimensional so take implicit time parameter for example
def visualize_all(X_1D, Y, X_train_1D, Y_train, X_test_1D, mean_prediction, std_prediction):
    # , linestyle="dotted")
    plt.plot(X_1D, Y, label=r"true data", color="tab:blue")
    plt.plot(X_test_1D, mean_prediction,
             label="mean prediction", color="tab:pink")
    plt.fill_between(
        X_test_1D.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        color="tab:green",
        alpha=0.5,
        label=r"95% confidence interval (total uncertainty)",
    )
    plt.scatter(X_train_1D, Y_train, color='red')
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    _ = plt.title("Gaussian process regression")
    plt.show()


def visualize_test(X_test_1D, Y_test, mean_prediction, std_prediction, epistemic_std=None, figure_num=1, title="MIAU"):
    plt.figure(figure_num)
    plt.plot(X_test_1D, Y_test, label=r"true data",
             color="tab:blue")  # , linestyle="dotted")
    plt.plot(X_test_1D, mean_prediction,
             label="mean prediction", color="tab:pink")
    plt.fill_between(
        X_test_1D.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        color="tab:green",
        alpha=0.5,
        label=r"95% confidence interval",
    )
    if epistemic_std is not None:
        plt.fill_between(
            X_test_1D.ravel(),
            mean_prediction - 1.96 * epistemic_std,
            mean_prediction + 1.96 * epistemic_std,
            color="tab:blue",
            alpha=0.5,
            label=r"95% confidence interval epistemic",
        )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    _ = plt.title(title)
    plt.show()


def visualize_multiple(X_test_1D, Y_test, predictions, figure_num=1, title="MIAU", csvpath=None):
    plt.figure(figure_num)
    plt.plot(X_test_1D, Y_test, label=r"true data",
             color="black")  # , linestyle="dotted")
    if csvpath is not None:
        df = pd.DataFrame()
        df['x'] = X_test_1D
        df['y'] = Y_test

    for prediction in predictions:
        plt.plot(X_test_1D, prediction["mean"], label=prediction["name"] +
                 " mean prediction", color=prediction["color"])
        plt.fill_between(
            X_test_1D.ravel(),
            prediction["mean"] - 1.96 * np.sqrt(prediction["var"]),
            prediction["mean"] + 1.96 * np.sqrt(prediction["var"]),
            color=prediction["color"],
            alpha=0.5,
            label=prediction["name"] + r" 95% confidence interval",
        )
        if csvpath is not None:
            df[prediction["name"] + "_mean"] = prediction["mean"].to_numpy()
            df[prediction["name"] + "_upper_CI"] = (prediction["mean"] + 1.96 * np.sqrt(prediction["var"])).to_numpy()
            df[prediction["name"] + "_lower_CI"] = (prediction["mean"] - 1.96 * np.sqrt(prediction["var"])).to_numpy()
    
    df['lower_CI'] = df[[prediction["name"] + "_lower_CI" for prediction in predictions]].max(axis=1)
    df['upper_CI'] = df[[prediction["name"] + "_upper_CI" for prediction in predictions]].min(axis=1)

    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    _ = plt.title(title)


    # save csv
    if csvpath is not None:
        df.to_csv(csvpath, index=False)

    plt.show()


#  ▄▀▀ ▄▀▄ █   █ ██▄ █▀▄ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  ▀▄▄ █▀█ █▄▄ █ █▄█ █▀▄ █▀█  █  █ ▀▄▀ █ ▀█

def visualize_calibration(X_test_1D, Y_test, calibration_outputs_norm, calibration_outputs_cdf, CIs, index):
    """
    calibration_outputs: [(t_values, cdf, name)]
    CIs: list of confidence intervals, ordered
    cdf: cumulative distribution functions shape (t,n,1)
        with t as the number of points that define the PDF/CDF per sample of n,
        with n as the number of samples each one corresponding to a value in the means array,
    """
    # colormap
    normalized_points = np.linspace(0, 1, len(CIs))
    # You can replace 'viridis' with any other colormap like 'plasma', 'magma', etc.
    colormap = plt.cm.viridis
    colors = [colormap(1 - point) for point in normalized_points]

    #plot the uncalibrated and calibrated NORM data
    for i, (mean_prediction, std_prediction, name) in enumerate(calibration_outputs_norm):
        mean_prediction = np.array(mean_prediction).ravel()
        fig = plt.figure(i, figsize=(15, 5))
        # true function
        plt.plot(X_test_1D, Y_test, label=r"TROLL", color="green")
        # predicted mean
        plt.plot(X_test_1D, mean_prediction, label=r"Multi-fidelity prediction", color="blue")

        # plot CIs
        for j, CI in enumerate(CIs):
            alpha = 1.0 - CI
            plt.fill_between(
                X_test_1D.ravel(),
                mean_prediction - norm.ppf(1-alpha/2) * std_prediction,
                mean_prediction + norm.ppf(1-alpha/2) * std_prediction,
                color="blue",
                alpha = 0.2, 
                ec='None',
                label=str(int(100*CI)) + "\% confidence interval",
            )

        plt.title(name)
        plt.xlabel("simulation time, [s]") 
        plt.ylabel("traction force, [N]") 
        plt.tight_layout()
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.savefig('D://Vlad/plots_for_MuDaFuGP/uncalibr_' + str(i) + '_' + str(index) + '_' + '.pdf', bbox_inches='tight')
        plt.show()

    # plot the CDF calibrated data
    for i, (t_values, cdf, name) in enumerate(calibration_outputs_cdf):
        fig = plt.figure(i+len(calibration_outputs_norm), figsize=(15, 5))
        # true function
        plt.plot(X_test_1D, Y_test, label=r"TROLL", color="green")

        mean_calib, _ = cumulative_moments(t_values, cdf)
        mean_calib = np.squeeze(mean_calib)

        # predicted mean
        plt.plot(X_test_1D, mean_calib, label=r"Calibrated multi-fidelity prediction", color="blue")

        # plot CIs NOTE CIs don't have to be centered around the mean here necessarily bc the distribution isn't normal anymore
        n = mean_calib.shape[0]
        for j, CI in enumerate(CIs):
            alpha = 1 - CI
            lower = np.array(
                [np.interp(alpha/2, cdf[:, k, 0], t_values[:, k, 0]) for k in range(n)])
            upper = np.array(
                [np.interp(1-alpha/2, cdf[:, k, 0], t_values[:, k, 0]) for k in range(n)])
            plt.fill_between(
                X_test_1D.ravel(),
                lower,
                upper,
                color="blue",
                alpha = 0.2, 
                ec='None',
                label=str(int(100*CI)) + "\% confidence interval",
            )

        plt.title(name)   
        plt.xlabel("simulation time, [s]") 
        plt.ylabel("traction force, [N]") 
        plt.tight_layout()
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.savefig('D://Vlad/plots_for_MuDaFuGP/calibr_' + str(i) + '_' + str(index) + '_' + '.pdf', bbox_inches='tight')
        plt.show()

def visualize_calibration_map_Iso(iso_model, tikzpath=None, csvpath=None):
    # Obtain a dense range of x-values for prediction
    x_dense = np.linspace(0, 1, 1000)
    y_pred = iso_model._iso[0].predict(x_dense)

    # Plot
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x_dense, y_pred, label="Isotonic Regression")
    plt.plot([0, 1], [0, 1], 'k--', color='gray', label="identity map")

    plt.legend()
    plt.title("Isotonic Regression CDF Map")

    # save tikz figure
    if tikzpath is not None:
        tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=tikzpath)
        with open(tikzpath, "w") as open_file:
            open_file.write(tikz_fig)
    
    # save data to csv file
    if csvpath is not None:
        # Create a DataFrame
        df = pd.DataFrame(y_pred, columns=['y'])
        df.insert(0, 'x', x_dense)

        # Save the DataFrame to a CSV file
        df.to_csv(csvpath, index=False)

    plt.show()

def visualize_calibration_map_Beta(gpbeta_model, input_params, tikzpath=None, csvpath=None):
    fig = plt.figure(figsize=(10, 5))
    plt.plot([0, 1], [0, 1], 'k--', color='gray', label="Identity map")

    if csvpath is not None:
        df = pd.DataFrame()

    for i, (mu, sigma) in enumerate(input_params):
        t_beta, _, cdf_beta = gpbeta_model.transform((np.array([mu]), np.array([sigma])))
        cdf_beta = cdf_beta.flatten()
        cdf_test = norm.cdf(t_beta, loc=mu, scale=sigma).flatten()
        plt.plot(cdf_test, cdf_beta, label="\mu = " + str(mu) + ", \sigma = " + str(sigma))
        if csvpath is not None:
            df.insert(0, 'x_' + str(i), cdf_test)
            df.insert(0, 'y_' + str(i), cdf_beta)
    
    plt.title("Beta Calibration map")
    plt.legend()

    # save tikz figure
    if tikzpath is not None:
        tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=tikzpath)
        with open(tikzpath, "w") as open_file:
            open_file.write(tikz_fig)

    #save csv
    if csvpath is not None:
        df.to_csv(csvpath, index=False)
    
    plt.show()

def visualize_calibration_map_Normal(gpnorm_model, input_params, tikzpath=None, csvpath=None):
    x_values = np.linspace(-100,100,1000)
    fig = plt.figure(figsize=(10, 5))
    plt.plot([0, 1], [0, 1], 'k--', color='gray', label="Identity map")

    if csvpath is not None:
        df = pd.DataFrame()

    for i, (mu, sigma) in enumerate(input_params):
        std_norm = gpnorm_model.transform((np.array([mu]), np.array([sigma])))[0]
        cdf_test = norm.cdf(x_values, loc=mu, scale=sigma).flatten()
        cdf_norm = norm.cdf(x_values, loc=mu, scale=std_norm).flatten()
        plt.plot(cdf_test, cdf_norm, label="$μ = " + str(mu) + ", σ = " + str(sigma))
        if csvpath is not None:
            df.insert(0, 'x_' + str(i), cdf_test)
            df.insert(0, 'y_' + str(i), cdf_norm)

    plt.title("Normal Calibration map")

    # save tikz figure
    if tikzpath is not None:
        tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=tikzpath)
        with open(tikzpath, "w") as open_file:
            open_file.write(tikz_fig)

    if csvpath is not None:
        df.to_csv(csvpath, index=False)

    plt.show()



#  ▄▀▀ ▄▀▄ █▄ ▄█ █▀▄ █ █ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █ ▄▀▄ █     █ █ █▄ █ ▄▀▀ ██▀ █▀▄ ▀█▀ ▄▀▄ █ █▄ █ ▀█▀ ▀▄▀
#  ▀▄▄ ▀▄▀ █ ▀ █ █▀  ▀▄█  █  █▀█  █  █ ▀▄▀ █ ▀█ █▀█ █▄▄   ▀▄█ █ ▀█ ▀▄▄ █▄▄ █▀▄  █  █▀█ █ █ ▀█  █   █ 


def visualize_compstatistics(x, y_1, std_y_1=None,  label_1="y1", title="Title", tikzpath=None):
    fig = plt.figure()
    plt.plot(x, y_1, label=label_1, color="tab:blue")

    if std_y_1 is not None:
        plt.fill_between(x, y_1 - std_y_1, y_1 + std_y_1, color="tab:blue", alpha=0.5)

    plt.xscale('log')
    plt.legend()
    plt.title(title)

    # save tikz figure
    if tikzpath is not None:
        tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=tikzpath)
        with open(tikzpath, "w") as open_file:
            open_file.write(tikz_fig)

    plt.show()


def visualize_compstatistics_comparison(x, y_1, y_2, std_y_1=None, std_y_2=None, label_1="y1", label_2="y2", title="Title", tikzpath=None):
    fig = plt.figure()
    plt.plot(x, y_1, label=label_1, color="tab:blue")
    plt.plot(x, y_2, label=label_2, color="tab:pink")

    if std_y_1 is not None:
        plt.fill_between(x, y_1 - std_y_1, y_1 + std_y_1, color="tab:blue", alpha=0.5)
    if std_y_2 is not None:
        plt.fill_between(x, y_2 - std_y_2, y_2 + std_y_2, color="tab:pink", alpha=0.5)

    plt.xscale('log')
    plt.legend()
    plt.title(title)

    # save tikz figure
    if tikzpath is not None:
        tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=tikzpath)
        with open(tikzpath, "w") as open_file:
            open_file.write(tikz_fig)

    plt.show()


#some colors
dark = (0.015686275, 0.035294118, 0.149019608)
darkblue = (0.043137255, 0.11372549, 0.470588235)
lightgreen = (0.560784314, 0.819607843, 0.384313725)
lightblue = (0.294117647, 0.71372549, 0.839215686)


def visualize_math_iter(X_test_1D, Y_test, mean_prediction_math, var_prediction_math, mean_prediction_iter, var_prediction_iter, figure_num=1):
    plt.figure(figure_num)

    # true function
    plt.plot(X_test_1D, Y_test, label=r"Latent function",
             linestyle="dashed", color=dark)

    # # mathematical prediction
    # plt.plot(X_test_1D, mean_prediction_math,
    #          label="Mean prediction (mathematical)", color="black")

    # computational prediction
    plt.plot(X_test_1D, mean_prediction_iter,
             label="Mean prediction (approximate)", color=darkblue)

    # computational uncertainty
    plt.fill_between(
        X_test_1D.ravel(),
        mean_prediction_iter - 1.96 *
        np.sqrt(var_prediction_iter - var_prediction_math),
        mean_prediction_iter + 1.96 *
        np.sqrt(var_prediction_iter - var_prediction_math),
        color=lightgreen,
        alpha=0.5,
        label=r"95% confidence interval, computational uncertainty",
    )

    # mathematical uncertainty shade 1
    plt.fill_between(
        X_test_1D.ravel(),
        mean_prediction_iter - 1.96 * np.sqrt(var_prediction_iter - var_prediction_math),
        mean_prediction_iter - 1.96 * np.sqrt(var_prediction_iter),
        color=lightblue,
        alpha=0.5,
        label=r"95% confidence interval, mathematical uncertainty",
    )
     # mathematical uncertainty shade 2
    plt.fill_between(
        X_test_1D.ravel(),
        mean_prediction_iter + 1.96 * np.sqrt(var_prediction_iter - var_prediction_math),
        mean_prediction_iter + 1.96 * np.sqrt(var_prediction_iter),
        color=lightblue,
        alpha=0.7,
    )

    # # mathematical uncertainty line
    # plt.plot(X_test_1D, mean_prediction_math - 1.96 *
    #          np.sqrt(var_prediction_math), color="grey", linestyle="dotted")
    # plt.plot(X_test_1D, mean_prediction_math + 1.96 *
    #          np.sqrt(var_prediction_math), color="grey", linestyle="dotted")

    #plt.legend()
    plt.xlabel("time")
    plt.ylabel("traction force")
    plt.ylim=(-12.5, 7.5)
    plt.show()



#  ▄▀▄ ▀█▀ █▄█ ██▀ █▀▄
#  ▀▄▀  █  █ █ █▄▄ █▀▄

def visualize_var_decomposition(X_test_1D, var_mathematical, var_computational):
    plt.fill_between(
        X_test_1D.ravel(),
        var_mathematical,
        color=lightblue,
        alpha=0.7,
        label=r"mathematical uncertainty",
    )
    plt.fill_between(
        X_test_1D.ravel(),
        var_mathematical,
        var_mathematical + var_computational,
        color=lightgreen,
        alpha=0.5,
        label=r"compuational uncertainty",
    )
    #plt.legend()
    plt.xlabel("time")
    plt.ylabel("predicted variance")
    plt.ylim=(0, 7)
    plt.show()


# Feature Selection visualization
descriptions = ["pos 1", "pos 2", "velocity 0", "velocity 1", "velocity 2",
                "angular velocity 0", "angualr velocity 1", "angular velocity 2", "gravity 1", "gravity 2"]

descriptions_2 = ["pos 1", "pos 2", "velocity 0", "velocity 1", "velocity 2",
                  "angular velocity 0", "angualr velocity 1", "angular velocity 2", "gravity angle"]


def visualize_feature_selection(ranks, sort=False, labels=None, title=""):
    _labels = labels if not (labels is None) else descriptions if len(ranks) == len(
        descriptions) else descriptions_2
    if sort:
        combined = list(zip(_labels, ranks))
        combined.sort(key=lambda x: x[1], reverse=True)
        labels_sorted, ranks_sorted = zip(*combined)
        plt.bar(labels_sorted, ranks_sorted, color='skyblue')
    else:
        plt.bar(_labels, ranks, color='skyblue')
    plt.xticks(rotation=90)
    _ = plt.title(title)
    plt.show()


def visualize_feature_selection_color(ranks, colors, sort=False, labels=None, title=""):
    _labels = labels if labels is not None else descriptions if len(
        ranks) == len(descriptions) else descriptions_2
    if sort:
        combined = list(zip(_labels, ranks, colors))
        combined.sort(key=lambda x: x[1], reverse=True)
        labels_sorted, ranks_sorted, values_sorted = zip(*combined)
        plt.bar(labels_sorted, ranks_sorted,
                color=plt.cm.viridis(values_sorted))
    else:
        plt.bar(_labels, ranks, color=plt.cm.viridis(colors))
    plt.xticks(rotation=90)
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Color Legend')
    _ = plt.title(title)
    plt.show()


def visualize_feature_selection_color_2(ranks, colors, sort=False, labels=None, color_description="", title=""):
    _labels = labels if labels is not None else descriptions if len(
        ranks) == len(descriptions) else descriptions_2
    if sort:
        combined = list(zip(_labels, ranks, colors))
        combined.sort(key=lambda x: x[1], reverse=True)
        labels_sorted, ranks_sorted, colors_sorted = zip(*combined)
        bars = plt.bar(labels_sorted, ranks_sorted,
                       color=plt.cm.viridis(normalize(colors_sorted, colors)))
    else:
        bars = plt.bar(_labels, ranks, color=plt.cm.viridis(
            normalize(colors, colors)))

    plt.xticks(rotation=90)
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(
        vmin=min(colors), vmax=max(colors))), label=color_description)
    _ = plt.title(title)
    plt.show()


def normalize(values, original_values):
    return [(v - min(original_values)) / (max(original_values) - min(original_values)) for v in values]


def visualize_matrix(data, descriptions, title=""):
    plt.imshow(data, cmap='viridis')
    plt.xticks(range(data.shape[1]), descriptions)
    plt.yticks(range(data.shape[0]), descriptions)
    plt.xticks(rotation=90)
    plt.colorbar()
    _ = plt.title(title)
    plt.show()
