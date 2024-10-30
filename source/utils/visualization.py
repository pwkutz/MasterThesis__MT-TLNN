import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
#mpl.rcParams.update(mpl.rcParamsDefault)
#mpl.rcParams['text.usetex'] = True

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
# plt.style.use('ggplot')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20

rc('font', size=BIGGER_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

desired_length = 50
window = 20
sampling_rate_hf_train = 0.005

def visual_on_synthetic():
    for j in range(10): # for test set use TROLL-derived test-runs (they will be also used as the test set for entire MF model evaluation. is it ok??)
        X_test = X_lf[(how_many_points + (j*201)):(how_many_points + (j*201)+201), :]
        X_test = ((X_test-min_lf)/(max_lf-min_lf))[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]].copy()
        mean, var = nargdp.predict(X_test)
        plt.plot(mean, label = "MF surrogate") 
        #plt.plot(train_test_dataset_from_troll.dataset['troll'].train_test_dataset['test']['y'][j][:, index], label = "troll simulation")
        smoothed_test_y = np.array(y_lf[(how_many_points + (j*201)):(how_many_points + (j*201)+201), output_index]).ravel()
        plt.plot(smoothed_test_y, label = "scm simulation")
        plt.legend()
        plt.show()

def visual_on_experiments(mf_model, single_gp, y_scm_test, X_hf_test, y_hf_test, output_index, savefig = False):
    for i in range(len(X_hf_test)):
        #X_hf_test_scaled = ((X_hf_test[i]-min_hf)/(max_hf-min_hf))[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]].copy()
        test_run = X_hf_test[i]
        mean, var = mf_model.predict(test_run)
        mean_singl, var_singl = single_gp.predict(test_run)
        plt.figure(figsize=(15, 5))
        sampling_rate_hf = 10 # hardcoded in data generation
        plt.plot(np.linspace(0, len(test_run)*sampling_rate_hf/60, len(mean)), mean, label = "multi-fidelity surrogate") 
        plt.fill_between(np.linspace(0, len(test_run)*sampling_rate_hf/60, len(mean)),
                np.array(mean).ravel() - np.sqrt(np.abs(var)).ravel(), 
                np.array(mean).ravel() + np.sqrt(np.abs(var)).ravel(), 
                alpha = 0.2, ec='None', label='multi-fidelity surrogate 95\\% confidence')
        plt.plot(np.linspace(0, len(test_run)*sampling_rate_hf/60, len(mean_singl)), mean_singl, label = "single fidelity prediction")
        plt.fill_between(np.linspace(0, len(test_run)*sampling_rate_hf/60, len(mean_singl)),
                np.array(mean_singl).ravel() - np.sqrt(np.abs(var_singl)).ravel(),
                np.array(mean_singl).ravel() + np.sqrt(np.abs(var_singl)).ravel(),
                alpha = 0.2, ec='None', label='single fidelity 95\\% confidence')
        plt.plot(np.linspace(0, len(test_run)*sampling_rate_hf/60, len(y_hf_test[i][:, output_index])), y_hf_test[i][:, output_index], label = "TROLL")
        scm_y = y_scm_test[i][:, output_index]
        plt.plot(np.linspace(0, len(test_run)*sampling_rate_hf/60, len(scm_y)), scm_y, label = "SCM")
        plt.xlabel("simulation time, [s]") #, fontsize = 20)
        plt.ylabel("traction force, [N]") #, fontsize = 20)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.grid(True)
        #plt.legend(loc='upper center', fontsize=20, ncol=3)
        if savefig:
            plt.savefig('D://Vlad/plots_for_MuDaFuGP/preds_vis/preds_vis' + str(i) + '.pdf', bbox_inches='tight')

'''def visual_learned_surface_normal(surrogate_model, X_train, y_train, X_test, y_test, output_index):
    ## make it a more universal, a class
    sink_input = np.linspace(0, 1.2, 20)
    vel_input = np.linspace(0, 1.2, 20)

    sink_full_cover, vel_full_cover = np.meshgrid(sink_input, vel_input)
    #full_cover_test = np.concatenate([sink_full_cover.reshape(-1, 1), vel_full_cover.reshape(-1, 1)], axis=1)
    # normal force: 6 features, [0, 2*, 3, 5*, 9, 11]
    full_cover_test_w_other_fixed = np.concatenate([np.repeat(0.5, 400)[:, None], 
                                                    sink_full_cover.reshape(-1, 1), 
                                                    np.repeat(0.5, 400)[:, None], 
                                                    vel_full_cover.reshape(-1, 1), 
                                                    np.repeat(np.repeat(0.5, 400)[:, None], 2, axis=1)], axis=1)
    spred_mean, _ = surrogate_model.predict(full_cover_test_w_other_fixed)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(sink_full_cover.reshape(-1, 1)[:, 0], vel_full_cover.reshape(-1, 1)[:, 0], spred_mean, alpha=0.2)

    xs = X_test[:, 1] # actual 2 
    ys = X_test[:, 3] # actual 5
    zs = y_test[:, output_index]
    ax.scatter(xs, ys, zs, marker='o', c="red")

    xd = X_train[::5, 1] # actual 2
    yd = X_train[::5, 3] # actual 5
    zd = y_train[::5, output_index]
    ax.scatter(xd, yd, zd, marker='o', c="green")

    ax.set_xlabel('sinkage')
    ax.set_ylabel('vertical velocity')
    ax.set_zlabel('Output index: ' + str(output_index))'''

def visual_learned_surface(surrogate_model, dataset_train, X_test, y_test, output_index, vis_indices):
    '''
    Visualize in 3D the training input space and the mapping function. 
    Takes as input and output indeces which you want to visualize.
    '''
    # show new positions (after filtering out 0-variance columns) of the features we want to visualize
    selected_indices = [i for i, x in enumerate(dataset_train.lf_norm.selected_columns) if x]
    pos_1 = selected_indices.index(vis_indices[0])
    pos_2 = selected_indices.index(vis_indices[1])

    ## make a unique variables for vis_indices, in order to check how the model learns variability in the space spanned by these features
    input0 = np.linspace(0, 1.2, 20)
    input1 = np.linspace(0, 1.2, 20)
    input0_full_cover, input1_full_cover = np.meshgrid(input0, input1)

    ## create an input matrix for the visualization
    n_columns = len(selected_indices)
    full_cover_test_w_other_fixed = np.full((400, n_columns), 0.5) # 400 and 0.5 are arbitrarily chosen, idk...
    full_cover_test_w_other_fixed[:, pos_1] = input0_full_cover.reshape(1, -1)[0]
    full_cover_test_w_other_fixed[:, pos_2] = input1_full_cover.reshape(1, -1)[0]

    spred_mean, _ = surrogate_model.predict(full_cover_test_w_other_fixed)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(input0_full_cover.reshape(-1, 1)[:, 0], 
               input1_full_cover.reshape(-1, 1)[:, 0],
               spred_mean, alpha=0.2)

    xs = X_test[:, pos_1] 
    ys = X_test[:, pos_2] 
    zs = y_test[:, output_index]
    ax.scatter(xs, ys, zs, marker='o', c="red", alpha=0.2)

    xd = dataset_train.X_lf_train[::2, pos_1] 
    yd = dataset_train.X_lf_train[::2, pos_2]
    zd = dataset_train.y_lf_train[::2, output_index]
    ax.scatter(xd, yd, zd, marker='o', c="green", alpha=0.2)

    ax.set_xlabel('Input index: ' + str(vis_indices[0]))
    ax.set_ylabel('Input index: ' + str(vis_indices[1]))
    ax.set_zlabel('Output index: ' + str(output_index))