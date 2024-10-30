class AbstrTerMechDataset:
    def __init__(self, train_test_dataset, simulation_name, train_runs_indeces, test_runs_indeces):
        self.simulation_name = simulation_name
        self.feature_names = ["pos1", "pos2", "pos3", "vel1", "vel2", "vel3", "ang_vel1", "ang_vel2", "ang_vel3", "norm1", "norm2", "norm3"]
        self.outputs_names = ["trac_force", "y_force", "z_force", "x_torque", "y_torque", "z_torque"]

        train = {'X': train_test_dataset[0],
                 'y': train_test_dataset[1],
                 'surface': train_test_dataset[2]}
        test = {'X': train_test_dataset[3],
                'y': train_test_dataset[4],
                'surface': train_test_dataset[5]}
        
        self.train_test_dataset = {"train": train, "test": test}
                                   
        self.train_runs_indeces = train_runs_indeces
        self.test_runs_indeces = test_runs_indeces
        self.num_train_runs = len(self.train_runs_indeces)
        self.num_test_runs = len(self.test_runs_indeces)
        #self.num_train = train_test_dataset[0].shape[0]
        self.num_test = len(train_test_dataset[2])
        #self.num_features = train_test_dataset[0].shape[1]

    def get_info():
        print("Info: ...")
