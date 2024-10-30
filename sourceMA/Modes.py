import sys

from sourceMA.NNL import CNN, TransferLearning, cross_validation
from sourceMA.Autoencoder import AE
from sourceMA import data_loader
from sourceMA.nn import NN

def Test_and_Train(config):

    '''execute Transfer Learning + predict test data'''

    (feat__train__HFLF, target__train__HF__HFLF, target__train__LF__HFLF,
    feat__test__HFLF, target__test__HF__HFLF, target__test__LF__HFLF) = data_loader.load_TrainData(config, config["file__train__HF"])
    (feat__train__LF, target__train__LF,
     feat__test__LF, target__test__LF) = data_loader.load_TrainData(config, config["file__train__LF"])



    AE_model = NN(config, "AE")
    NNL_model = NN(config, "NNL")
    #AE_model.plot_stats__nn()

    AE_model.SourceTraining( "AE", config, target__train__LF, target__train__LF) # SourceTrain + Freeze + Save
    NNL_model.SourceTraining( "NNL", config, feat__train__LF, target__train__LF)

    AE_model.load_model(config, "AE")
    NNL_model.load_model(config, "NNL")
    MFTLNN_model = NN(config=config, type="MFTLNN", AE = AE_model, NNL = NNL_model)
    MFTLNN_model.TaskTraining("MFTLNN", config, feat__train__HFLF, target__train__HF__HFLF, target__train__LF__HFLF)

    #AE_model.analysis(config, "AE", target__train__LF, target__train__LF, target__test__LF, target__test__LF)
    #NNL_model.analysis(config, "NNL", feat__train__LF, target__train__LF, feat__test__LF, target__test__LF)
    MFTLNN_model.analysis(config, "MFTLNN", [feat__train__HFLF, target__train__LF__HFLF], target__train__HF__HFLF,
                          [feat__test__HFLF, target__test__LF__HFLF], target__test__HF__HFLF)

    #############################################

    sys.exit()

    '''training'''

    TransferLearning(CNN_model, config,
                        feat__train__HF, target__train__HF,
                        feat__train__LF, target__train__LF)

    CNN_model.model.save(data_loader.resource_path(config["checkpoint_path"]))

    '''plot+analyse'''

    CNN_model.load_model(config, "AE")
    CNN_model.predicts(test_features)
    CNN_model.analyse(train_features__HF, train_target__HF, test_features, test_target)






def PerformanceCheck(config):

    features__HF, target__HF, features__LF, target__LF = data_loader.load_CommonDataset(config) # get LF and HF data
    average_acc, average_loss, file__maxAccuracy = cross_validation(config, # K-Fold Cross-Validation
                                                                    features__HF, target__HF,
                                                                    features__LF, target__LF)

    print("Average (Test)-Accuracy:", average_acc)
    print("Average (Test)-Loss:", average_loss)
    print("Model with optimal (Test-)Accuracy:", file__maxAccuracy)


def safety_check(config):

    if (config["mode"]["TestAndTrain"] is True) & (config["mode"]["PerformanceCheck"] is True):
        print("Mode Error: 'TestAndRun' and 'PerformanceCheck' are on")
        sys.exit()
    elif (config["mode"]["TestAndTrain"] is not True) & (config["mode"]["PerformanceCheck"] is not True):
        print("Mode Error: 'TestAndRun' and 'PerformanceCheck' are off")
        sys.exit()


