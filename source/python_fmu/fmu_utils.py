from fmpy import read_model_description, extract, instantiate_fmu
from fmpy.fmi2 import FMU2Slave
from fmpy.util import plot_result, download_file, download_test_file
from fmpy.fmi1 import fmi1OK

from tqdm import tqdm
import numpy as np
import h5py
import glob

# for FMU errors
def log_message(componentEnvironment, instanceName, status, category, message):
    if status == fmi1OK:
        pass  # do nothing
    else:
        print(message.decode('utf-8'))

def intialize_terra_fmu(fmu_filename = 'D://Vlad/work/ContactDynamics_ContactModels_TerRA_TerRATwoFrame_0FMU.fmu', output_feature = 'y[1]'): 
    # options: 'D://Vlad/work/scm_fmu_flat.fmu'

    # optimized for the timestep 1e-3
    model_description = read_model_description(fmu_filename)
    vrs = {}
    for variable in model_description.modelVariables:
        vrs[variable.name] = variable.valueReference
    vr_inputs = [vrs[x] for x in vrs.keys() if "u["==x[0:2]] # standard inputs, same for the SCM and TerRA FMUs 
    vr_outputs = [vrs[output_feature]] # traction force
    unzipdir = extract(fmu_filename)
    fmu = FMU2Slave(guid=model_description.guid,
                unzipDirectory=unzipdir,
                modelIdentifier=model_description.coSimulation.modelIdentifier,
                instanceName='instance1')
    fmu.instantiate(loggingOn=False)
    fmu.setupExperiment(startTime=0)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()
    return fmu, vr_inputs, vr_outputs

def call_fmu(input_vec, fmu, vr_inputs, vr_outputs):
    f_x = []
    #print(input_vec.shape[0], input_vec.shape[1])
    for i in range(input_vec.shape[0]):
        value_inputs = input_vec[i]
        fmu.setReal(vr_inputs, value_inputs)
        fmu.doStep(currentCommunicationPoint=i*0.001, communicationStepSize=0.001)
        f_x.append(fmu.getReal(vr_outputs))
    return np.array(f_x)

def fmu_close(fmu):
    fmu.terminate()
    fmu.freeInstance()

def lf_fmu(x, intialize_fmu = intialize_terra_fmu):
    ## transfer somehow the inner state of the terra
    fmu, vr_inputs, vr_outputs = intialize_fmu()
    # determune which index "x" has 
    # Can I always just upload additionally all previous steps of the current input x? 
    #indx = np.argwhere(X_train, x)
    output = []
    if sum(np.all(train_scm == x[0], axis=1)) > 0:
        index = np.where(np.all(train_scm == x[0], axis=1))[0][0]
        for element in x:
            output.append(call_fmu(np.vstack((train_scm[:index], element)), fmu, vr_inputs, vr_outputs)[-1])
    else:
        index = np.where(np.all(train_scm_test == x[0], axis=1))[0][0]
        for element in x:
            output.append(call_fmu(np.vstack((train_scm_test[:index], element)), fmu, vr_inputs, vr_outputs)[-1])
    return output # or X_train[:i]

def lf_fmu_grad(x, intialize_fmu = intialize_terra_fmu):
    print("i`m in lf grad")
    fmu, vr_inputs, vr_outputs = intialize_fmu()
    delta = 1e-5
    f_x_plus = []
    f_x_minus = []

    if sum(np.all(train_scm == x[0], axis=1)) > 0:
        index = np.where(np.all(train_scm == x[0], axis=1))[0][0]
        for element in x:
            if index == 0:
                with_last_X_changed_plus = np.array([train_scm[index, :] + delta]) 
                with_last_X_changed_minus = np.array([train_scm[index, :] - delta]) 
            else:
                with_last_X_changed_plus = np.vstack((train_scm[:index], element + delta)) 
                with_last_X_changed_minus = np.vstack((train_scm[:index], element - delta)) 
            fmu, vr_inputs, vr_outputs = intialize_fmu()
            changed_last_fmu_out_plus = call_fmu(with_last_X_changed_plus, fmu, vr_inputs, vr_outputs)[-1]
            fmu_close(fmu)
            fmu, vr_inputs, vr_outputs = intialize_fmu()
            changed_last_fmu_out_minus = call_fmu(with_last_X_changed_minus, fmu, vr_inputs, vr_outputs)[-1]
            fmu_close(fmu)
            fmu, vr_inputs, vr_outputs = intialize_fmu()
            fmu_close(fmu)
            ############# FIND HOW TO RESET FMU W\O FULL INITIALIZATION AND CLOSE #########################
            f_x_plus.append(changed_last_fmu_out_plus)
            f_x_minus.append(changed_last_fmu_out_minus)
        f_x_plus = np.array(f_x_plus)
        f_x_minus = np.array(f_x_minus)
        grad_x = abs(f_x_plus - f_x_minus)/(2*delta)
        
    else:
        index = np.where(np.all(train_scm_test == x[0], axis=1))[0][0]
        for element in x:
            if index == 0:
                with_last_X_changed_plus = np.array([train_scm_test[index, :] + delta]) 
                with_last_X_changed_minus = np.array([train_scm_test[index, :] - delta]) 
            else:
                with_last_X_changed_plus = np.vstack((train_scm_test[:index], x + delta)) 
                with_last_X_changed_minus = np.vstack((train_scm_test[:index], x - delta)) 
            fmu, vr_inputs, vr_outputs = intialize_fmu()
            changed_last_fmu_out_plus = call_fmu(with_last_X_changed_plus, fmu, vr_inputs, vr_outputs)[-1]
            fmu_close(fmu)
            fmu, vr_inputs, vr_outputs = intialize_fmu()
            changed_last_fmu_out_minus = call_fmu(with_last_X_changed_minus, fmu, vr_inputs, vr_outputs)[-1]
            fmu_close(fmu)
            fmu, vr_inputs, vr_outputs = intialize_fmu()
            fmu_close(fmu)
            ############# FIND HOW TO RESET FMU W\O FULL INITIALIZATION AND CLOSE #########################
            f_x_plus.append(changed_last_fmu_out_plus)
            f_x_minus.append(changed_last_fmu_out_minus)
        f_x_plus = np.array(f_x_plus)
        f_x_minus = np.array(f_x_minus)
        grad_x = abs(f_x_plus - f_x_minus)/(2*delta)  

    return grad_x #output

#Adjusting HDF5 file with auxiliary variables for the SCM FMU
def adjust_scm_data_for_fmu():
    filename = "D:/Vlad/work/scm_from_troll.h5"
    data = h5py.File(filename, 'a')
    out_ = np.array(data["aux"])
    tmp = []
    for i in tqdm(range(len(out_))):
        tmp.append(np.concatenate((np.array(out_[i][0])[np.newaxis], out_[i][1])))
    data.create_dataset('/aux_mod', data=np.array(tmp))
    data.close()
    import scipy.io
    name = 'scm_output.mat'
    scipy.io.savemat(name, mdict={'aux': np.array(tmp)})

if __name__ == "__main__":


    vrs = {}
    for variable in model_description.modelVariables:
        vrs[variable.name] = variable.valueReference

    vr_inputs = [vrs[x] for x in vrs.keys() if "u["==x[0:2]] # standard inputs
    #vr_inputs.append(vrs['elevationMap.surface.RunID'])

    vr_outputs = [vrs['wheel_inside_fmu.datagathering.output_values[1]']] # traction force
    unzipdir = extract(fmu_filename)
    fmu = FMU2Slave(guid=model_description.guid,
                unzipDirectory=unzipdir,
                modelIdentifier=model_description.coSimulation.modelIdentifier,
                instanceName='instance1')
    fmu.instantiate(loggingOn=False)
    fmu.setupExperiment(startTime=0)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    scm_ids = np.array([])
    for troll_id in tqdm(train): 
        # load troll data (hifi)
        troll_run_name = glob("H:/NOT_SAVED_TO_BACKUP/TROLL_data/vlad/"+"*"+str(troll_id)+"*")[0]
        troll_run = troll.load_troll_data(troll_run_name, mode="analysis")
        troll_dataset = troll.transform_run(troll_run)
        # filter out robotic arm acceleration
        a_tcp_rotated = arm_acceleration(troll_run)
        # compensate troll forces with the robotic arm acceleration
        troll_force_current = troll_dataset[:, 14] + 20.97*a_tcp_rotated[:, 0] # traction force
        troll_X_current = troll_dataset[:, 1:14]
        runtime = troll_dataset[-1, 0] # how long was the run?
        start_time = [x for x in analyazable_starttime if troll_id in analyazable_starttime[x]][0]
        scm_ids = np.concatenate((scm_ids, np.repeat(troll_id, round(runtime-start_time)*1-2)))

    train = scm_X[:, 12:]

    #f_x = []
    #print(input_vec.shape[0], input_vec.shape[1])
    for i in tqdm(range(train.shape[0])):
        value_inputs = np.concatenate((train[i], np.array([scm_ids[i]])))
        fmu.setReal(vr_inputs, value_inputs)
        fmu.doStep(currentCommunicationPoint=i*0.001, communicationStepSize=0.001)
        #f_x.append(fmu.getReal(vr_outputs))

        #READ FROM OUT.h5 TRACTION FORCE

    ################## COMPARE SCM MODELICA OUTPUT AND SCM-FMU
    filename = "C:/Users/fedi_vl/Documents/work/all_runs_from_troll.h5"
    data = h5py.File(filename, 'r')
    scm = data[str(4339)]['scm']
    scm_input = np.array([x[1][:12] for x in scm["input"]])
    scm_fmu_input = np.array([x[1] for x in scm["aux"]])
    scm_force = np.array([x[1][0] for x in scm["output"]])
    f_x = []
    vr_outputs = [vrs['wheel_inside_fmu.datagathering.output_values[1]']]
    for i in tqdm(range(scm_fmu_input.shape[0])):
        value_inputs = scm_fmu_input[i]
        fmu.setReal(vr_inputs, value_inputs)
        fmu.doStep(currentCommunicationPoint=i*0.001, communicationStepSize=0.001)
        f_x.append(fmu.getReal(vr_outputs))
    
    plt.figure(figsize=(15,10))
    plt.plot(f_x, label = "FMU")
    plt.plot(scm_force, label = "SCM")
    plt.legend()