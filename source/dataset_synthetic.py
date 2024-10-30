class dataset_synthetic:
    def __initialization__(self, train_share = 1):
        ## TODO: transfer synthetic dataset from "validation" class to this one ##
        self.scm_synthetic = None
        self.terra_synthetic = None

    def info():
        print("Data generated from synthetic SCM and TerRA simulations. \n TerRA and SCM datasets were created on TROLL trajectories, using TrollInput from ContactDynamics library. Modified version of TrollInput which was used here could be found on dev-vlad branch. \n Numerical solver: Rkfix4 \n Parameters used in Modelica in order to adjust generated TerRA and SCM data to TROLL data: \n z_offset = 0.0115 for SCM amd 0.0065 for TerRA. \n scm_step = 0.03 \n overall modelica solver step \n To check optimization or optimize data generation hyperparameters look in 'optimization.ipynb' \n To validate the data, call 'data_validation'")