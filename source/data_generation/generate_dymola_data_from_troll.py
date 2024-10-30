import sys
import numpy as np
sys.path.append("D://Vlad/Dymola_2022x_win/Dymola 2022x/Modelica/Library/python_interface/dymola.egg") # append path to the modelica's interface
from dymola.dymola_interface import DymolaInterface
from dymola.dymola_exception import DymolaException
from tqdm import tqdm
from time import time
import scipy.io
import h5py
import logging
from .. import troll_to_modelica
from glob import glob
import json

work = "D:/Vlad/work/work_python"

def init_dymola(path,work,libs):
    dymola = DymolaInterface(path)
    for lib in libs:
        dymola.openModel(lib)
    dymola.cd(work)    
    return dymola

def run_scm(allruns, stopTime, troll_id, bump_flag, dymola):
    # TODO: optimize for every run separatelly
    modelica_stepsize = 0.03 
    z_offset = 0.0115 
    scm_step = 0.01 # was 0.02... from optimization?
    names = ["trollInput_A.r_offset[3]", "configurator_SCM.contactDynamicsTimeStep", "configurator_SCM.soilUpdateTimeStep", "id"]
    values = [z_offset, scm_step, 5*scm_step, troll_id]
    
    if bump_flag:
        script_name = "Mufi.Experiments.TrollDataTest_to_scm_bump"
    else:
        script_name = "Mufi.Experiments.TrollDataTest_to_scm"

    result, values = dymola.simulateMultiResultsModel(script_name,
                                                      stopTime=stopTime, 
                                                      numberOfIntervals=10000, 
                                                      method="Rkfix4",
                                                      initialNames=names,  
                                                      initialValues=[values], 
                                                      fixedstepsize=modelica_stepsize,
                                                      resultFile='tmpout')

    if not result:
        print(dymola.getLastErrorLog())

    filename = work + '/scm_from_troll.h5'
    scm = h5py.File(filename, 'r')
    # append scm run to the final file
    allruns.create_dataset('/' + str(troll_id) + '/scm/input', data=scm["input"])
    allruns.create_dataset('/' + str(troll_id) + '/scm/output', data=scm["output"])
    allruns.create_dataset('/' + str(troll_id) + '/scm/aux', data=scm["aux"])

def run_terra(allruns, stopTime, troll_id, bump_flag, dymola):
    # TODO: optimize for every run separatelly
    modelica_stepsize = 0.001
    z_offset = 0.0065
    names = ["trollInput_A.r_offset[3]", "id"]
    values = [z_offset, troll_id]

    if bump_flag:
        script_name = "Mufi.Experiments.TrollDataTest_to_terra_bump"
    else:
        script_name = "Mufi.Experiments.TrollDataTest_to_terra"

    result, values = dymola.simulateMultiResultsModel(script_name, 
                                                      stopTime=stopTime, 
                                                      numberOfIntervals=10000, 
                                                      method="Rkfix4",
                                                      ## can't transfer python booleans to modelica's booleans, therefore four separate modelica scripts :(
                                                      initialNames=names,  
                                                      initialValues=[values], 
                                                      fixedstepsize=modelica_stepsize,
                                                      resultFile='tmpout')

    if not result:
        logging.error(dymola.getLastErrorLog())
    filename = work + '/terra_from_troll.h5'
    terra = h5py.File(filename, 'r')
    # append terra run to the final file
    allruns.create_dataset('/' + str(troll_id) + '/terra/input', data=terra["input"])
    allruns.create_dataset('/' + str(troll_id) + '/terra/output', data=terra["output"])
    allruns.create_dataset('/' + str(troll_id) + '/terra/aux', data=terra["aux"])

############################## TO DO: Save runs not to one hdf file, but rather store each run separately ###############################

def generate_from_troll():
    green = "\033[32;20;53m"
    bar_format = f"{green}{{l_bar}}{{bar:50}} [{{elapsed}}]{{r_bar}}"
    ## append paths to all needed modelica scripts
    path = "D:/Vlad/Dymola_2022x_win/Dymola 2022x/bin64/Dymola.exe"
    lib = ["D:/Vlad/library/ContactDynamics/package.mo", "D:/Vlad/library/SR_Utilities/package.mo", "D:/Vlad/library/Visualization2/Modelica/Visualization2/package.mo",
        "D:/Vlad/library/Visualization2Overlays/package.mo", "D:/Vlad/library/mufi/package.mo", "D:/Vlad/library/HDF5/package.mo"]

    dymola = init_dymola(path,work,lib)
    allruns = h5py.File('D:/Vlad/data/runs_with_fixed_vertmes.h5', 'a')
    json_w_runs_id = 'parameters/runs_id.json'

    with open(json_w_runs_id) as data_file:
        runs = json.load(data_file)
    flat_troll_runs = np.array(runs["flat_runs"])
    bump_troll_runs = np.array(runs["bump_runs"])
    all_troll_runs = np.concatenate([flat_troll_runs, bump_troll_runs])
    
    new_bumpy_runs = [4836, 4837, 4838, 4839, 4840, 4841, 4842, 4843, 4844, 4845, 4846, 4847, 4848, 4849, 4850, 4851, 4852, 4853, 4854, 4855, 4856, 4857, 4858, 4859, 4860, 4861, 4862, 4863, 4864, 4865, 4866, 4867, 4868, 4869, 4870, 4871, 4872, 4872, 4873, 4874]

    for troll_id in tqdm(new_bumpy_runs, bar_format=bar_format, desc="troll runs"): #for troll_id in tqdm(all_troll_runs, bar_format=bar_format, desc="troll runs"):
        troll_run = glob("H:/NOT_SAVED_TO_BACKUP/TROLL_data/vlad/"+"*"+str(troll_id)+"*")[0]
        run = troll_to_modelica.load_troll_data(troll_run, mode="analysis")
        for_dymula = troll_to_modelica.transform_run(run)
        name = 'modelica_compatible.mat'
        #write the right .mat file to read it to the modelica's scm
        scipy.io.savemat(name, mdict={'myrun': for_dymula})
        stopTime = run['time'][-1]
        bump_flag = False
        if troll_id in bump_troll_runs:
            bump_flag = True
        if run == None:
                continue
        try:
            run_scm(allruns, stopTime, troll_id, bump_flag, dymola)
            run_terra(allruns, stopTime, troll_id, bump_flag, dymola)
        except:
            logging.warning("Restarting connection with Dymola...") # sometimes connection with dymola dies ...
            dymola = init_dymola(path,work,lib)
            run_scm(allruns, stopTime, troll_id, bump_flag, dymola)
            run_terra(allruns, stopTime, troll_id, bump_flag, dymola)
    allruns.close()