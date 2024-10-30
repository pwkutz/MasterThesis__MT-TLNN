#################################################################################################################################
#### find optimal parameters, like z-offset, scm soil update and modelica timestep for replication of TROLL data in Modelica ####
#################################################################################################################################
import sys
from tqdm import tqdm
from sklearn.metrics.regression import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

green = "\033[32;20;53m"
bar_format = f"{green}{{l_bar}}{{bar:50}} [{{elapsed}}]{{r_bar}}"

sys.path.append("D://Vlad/Dymola_2022x_win/Dymola 2022x/Modelica/Library/python_interface/dymola.egg")
#sys.path.append("D://Vlad/Dymola_2020/Modelica/Library/python_interface/dymola.egg")
from dymola.dymola_interface import DymolaInterface
from dymola.dymola_exception import DymolaException

def run_optimization(troll_id):
    path = "D:/Vlad/Dymola_2022x_win/Dymola 2022x/bin64/Dymola.exe"
    lib = ["D:/Vlad/library/ContactDynamics/package.mo", "D:/Vlad/library/SR_Utilities/package.mo", "D:/Vlad/library/Visualization2/Modelica/Visualization2/package.mo",
        "D:/Vlad/library/Visualization2Overlays/package.mo", "D:/Vlad/library/mufi/package.mo", "D:/Vlad/library/HDF5/package.mo"]
    work = "D:/Vlad/work/work_python"
    parameters_path = "D:/Vlad/mufintroll/parameters/"

    def init_dymola(path,work,libs):
        dymola = DymolaInterface(path)
        for lib in libs:
            dymola.openModel(lib)
        dymola.cd(work)    
        return dymola

    dymola = init_dymola(path,work,lib)

    overall_modelica_step_size = 0.03
    scm_timestep = 0.02
    scm_error_offset = {}

    with open(parameters_path + 'runs_id.json') as data_file:
            runs = json.load(data_file)
    troll_flat_runs = runs['flat_runs']

    with open(parameters_path + 'runs_stop_time.json') as data_file:
            troll_runs_stop_time = json.load(data_file)

    #for scm_timestep in tqdm(np.linspace(0.005, 0.0001, 30), desc="scm config step"):
    for offset in tqdm(np.linspace(0, 0.02, 20), desc="Z offset"):#, bar_format=bar_format, leave=True):
        #runs = {}
        #for troll_id in tqdm(troll_flat_runs, desc="troll runs"):#, bar_format=bar_format, leave=False):
        result, values = dymola.simulateMultiResultsModel("Mufi.Experiments.TrollDataTest_to_scm_example", stopTime=troll_runs_stop_time[str(troll_id)], 
                                                        numberOfIntervals=1000, 
                                                        method="Rkfix4",
                                                        ## can't transfer python booleans to modelica's booleans, therefore three separate modelica scripts :(
                                                        initialNames=["trollInput_A.r_offset[3]", "trollInput_A.trollFileID", 
                                                                        "configurator_SCM.contactDynamicsTimeStep", "configurator_SCM.soilUpdateTimeStep"],# "wheel_With_MLTM.mLTMFrame.gpr.mltm_step"], 
                                                        initialValues=[[offset, troll_id, scm_timestep, 5*scm_timestep]],# mltm_step_size]], # 0.010065326633165828 is optimal for scm, -0.03 for terra
                                                        resultNames=["A_sim_force[1]", "A_exp_force[1]", "wheel_With_MLTM.mLTMFrame.force[1]"],
                                                        fixedstepsize=overall_modelica_step_size,
                                                        resultFile='tmpout')
        output = pd.read_csv(work + "/multOut.csv", header=0, sep=";")
        output.loc[output.iloc[:, 0].isna(), "A_sim_force[1]"] = 0
        #runs[troll_id] = mean_squared_error(output.iloc[:-1, 0], output.iloc[:-1, 1])

        #scm_errors_config_step_size[scm_timestep] = runs
        #scm_error_offset[offset] = runs
        scm_error_offset[offset] = mean_squared_error(output.iloc[:-1, 0], output.iloc[:-1, 1])

    plt.figure(figsize=(25,7))

    plt.plot(np.linspace(0, 0.02, 20), [np.mean(list(scm_error_offset[x].values())) for x in scm_error_offset])
    plt.xticks(np.arange(0, 0.02, 0.0005), fontsize=8)
    plt.grid(linestyle='--')
    plt.xlabel("Z offset for SCM update")
    plt.ylabel("MSE of traction force, [N]")


    overall_modelica_step_size = 0.0001
    terra_error_offset = {}

    for offset in tqdm(np.linspace(0, 0.02, 20), desc="Z offset"):
        #runs = {}
        #for troll_id in tqdm(troll_flat_runs, desc="troll runs"):#, bar_format=bar_format, leave=False):
        result, values = dymola.simulateMultiResultsModel("Mufi.Experiments.TrollDataTest_to_terra_example", stopTime=troll_runs_stop_time[str(troll_id)], 
                                                        numberOfIntervals=1000, 
                                                        method="Rkfix4",
                                                        ## can't transfer python booleans to modelica's booleans, therefore three separate modelica scripts :(
                                                        initialNames=["trollInput_A.r_offset[3]", "trollInput_A.trollFileID"],
                                                        initialValues=[[offset, troll_id]],
                                                        resultNames=["A_sim_force[1]", "A_exp_force[1]", "wheel_With_MLTM.mLTMFrame.force[1]"],
                                                        fixedstepsize=overall_modelica_step_size,
                                                        resultFile='tmpout')
        output = pd.read_csv(work + "/multOut.csv", header=0, sep=";")
        output.loc[output.iloc[:, 0].isna(), "A_sim_force[1]"] = 0
        #runs[troll_id] = mean_squared_error(output.iloc[:-1, 0], output.iloc[:-1, 1])
        terra_error_offset[offset] = mean_squared_error(output.iloc[:-1, 0], output.iloc[:-1, 1])


    plt.figure(figsize=(25,7))

    plt.plot(np.linspace(0, 0.02, 20), [np.mean(list(scm_error_offset[x].values())) for x in scm_error_offset])
    plt.xticks(np.arange(0, 0.02, 0.0005), fontsize=8)
    plt.grid(linestyle='--')
    plt.xlabel("Z offset for SCM update")
    plt.ylabel("MSE of traction force, [N]")


    # Optimize the SCM config step (both contact and soilUpdate steps)
    overall_modelica_step_size = 0.0001
    scm_errors_config_step_size = {}
    #scm_timestep = 0.001
    z_offset = 0.017
    for scm_timestep in tqdm(np.linspace(0.01, 0.0001, 15), desc="scm config step", bar_format=bar_format):
        #runs = {}
        #for troll_id in tqdm(troll_flat_runs, desc="troll runs", bar_format=bar_format):
        result, values = dymola.simulateMultiResultsModel("Mufi.Experiments.TrollDataTest_to_scm_example", stopTime=troll_runs_stop_time[str(troll_id)], 
                                                        numberOfIntervals=1000, 
                                                        method="Rkfix4",
                                                        ## can't transfer python booleans to modelica's booleans, therefore three separate modelica scripts :(
                                                        initialNames=["trollInput_A.r_offset[3]", "trollInput_A.trollFileID", 
                                                                        "configurator_SCM.contactDynamicsTimeStep", "configurator_SCM.soilUpdateTimeStep"],# "wheel_With_MLTM.mLTMFrame.gpr.mltm_step"], 
                                                        initialValues=[[z_offset, troll_id, scm_timestep, 5*scm_timestep]],# mltm_step_size]], # 0.010065326633165828 is optimal for scm, -0.03 for terra
                                                        resultNames=["A_sim_force[1]", "A_exp_force[1]", "wheel_With_MLTM.mLTMFrame.force[1]"],
                                                        fixedstepsize=overall_modelica_step_size,
                                                        resultFile='tmpout')
        output = pd.read_csv(work + "/multOut.csv", header=0, sep=";")
        output.loc[output.iloc[:, 0].isna(), "A_sim_force[1]"] = 0
        #runs[troll_id] = mean_squared_error(output.iloc[:-1, 0], output.iloc[:-1, 1])
        scm_errors_config_step_size[scm_timestep] = mean_squared_error(output.iloc[:-1, 0], output.iloc[:-1, 1])

    return scm_error_offset, terra_error_offset, scm_errors_config_step_size