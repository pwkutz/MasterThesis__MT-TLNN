import sys
import os
from typing import List

import pandas as pd
import numpy as np
import ruamel.yaml
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
# from sourceMA import NNL
import pickle


def resource_path(relative_path):
    '''
    return total path for a given relative path
    total path is "pyinstaller conform"
    '''

    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def load_yaml(file_name: str):  # -> Union[Dict, None]:
    """
    Loads configuration setup from a yaml file

    :param file_name: name of the yaml file
    """

    # Use this to load your config file
    file_name = resource_path(file_name)

    with open(file_name, 'r') as stream:
        try:
            config = ruamel.yaml.round_trip_load(stream, preserve_quotes=True)
            return config
        except ruamel.yaml.YAMLError as exc:
            print(exc)
            return None


def show_image(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()


def remove_contentless_images__MultiFidelity(features, target__HF, target__LF):
    '''remove all images (LF/HF) which do not possess any content.
    Contentless images are image those pixels possess all the same value (e.g., 0)'''

    index__NonZero = [x for x in range(len(features)) if len(features.iloc[x].unique()) > 1]
    features = features.iloc[index__NonZero].reset_index(drop=True)
    target__HF = target__HF.iloc[index__NonZero].reset_index(drop=True)
    target__LF = target__LF.iloc[index__NonZero].reset_index(drop=True)

    return features, target__HF, target__LF


def remove_contentless_images(features, target):
    '''remove all images (LF/HF) which do not possess any content.
    Contentless images are image those pixels possess all the same value (e.g., 0)'''

    index__NonZero = [x for x in range(len(features)) if len(features.iloc[x].unique()) > 1]
    features = features.iloc[index__NonZero].reset_index(drop=True)
    target = target.iloc[index__NonZero].reset_index(drop=True)

    return features, target


def split__train_and_test__MultiFidelity(features, target__HF, target__LF):
    '''
    split existing HF dataset for training into train and test dataset
    Original test dataset consists out to 100 % out of content-free images (not usable)
    '''

    num__train = int(np.round(len(features) * 0.9))
    num__test = int(len(features) - num__train)
    all_samples = list(range(len(features)))

    index__test: list[int] = list(range(0, len(all_samples), int(len(all_samples) / num__test)))

    index__train = [x for x in all_samples if x not in index__test]

    feat__train = features[index__train]
    target__train__HF = target__HF.iloc[index__train]
    target__train__LF = target__LF.iloc[index__train]
    feat__test = features[index__test]
    target__test__HF = target__HF.iloc[index__test]
    target__test__LF = target__LF.iloc[index__test]

    return feat__train, target__train__HF, target__train__LF, feat__test, target__test__HF, target__test__LF


def split__train_and_test(features, target):
    '''
    split existing HF dataset for training into train and test dataset
    Original test dataset consists out to 100 % out of content-free images (not usable)
    '''

    num__train = int(np.round(len(features) * 0.9))
    num__test = int(len(features) - num__train)
    all_samples = list(range(len(features)))

    index__test = list(range(0, len(all_samples), int(len(all_samples) / num__test)))

    index__train = [x for x in all_samples if x not in index__test]

    feat__train = features[index__train]
    target__train = target.iloc[index__train]

    feat__test = features[index__test]
    target__test = target.iloc[index__test]

    return feat__train, target__train, feat__test, target__test


def preprocessing__MultiFidelity(window_size, features, target__HF, target__LF):
    '''preprocess features which are co-notated with several function values of different fidelities'''

    features, target__HF, target__LF = remove_contentless_images__MultiFidelity(features, target__HF, target__LF)
    features, target__HF, target__LF = apply__moving_average__MultiFidelity(window_size, features, target__HF, target__LF)
    features = features.T  # MinMaxScaler normalizes columns-wise only. But the flattened images are row-wise inside df
    features, target__HF, target__LF, scaler__target = normalize_data__MultiFidelity(features, target__HF,
                                                                                     target__LF)  # normalize each image separately

    features = pd.DataFrame(features).T  # re-arrange image DataFrame to old order where images are sorted row-wise.
    features = transform_to_scans(features)
    # show_image(features[0])
    target__HF = map(lambda x: x[0], target__HF)  # turn target from Numpy Array back to Pandas Series
    target__LF = map(lambda x: x[0], target__LF)
    target__HF = pd.Series(target__HF)
    target__LF = pd.Series(target__LF)

    (feat__train, target__train__HF, target__train__LF,
     feat__test, target__test__HF, target__test__LF) = split__train_and_test__MultiFidelity(features, target__HF,
                                                                                            target__LF)
    return (feat__train, target__train__HF, target__train__LF,
            feat__test, target__test__HF, target__test__LF, scaler__target)


def preprocessing__SingleFidelity(window_size, features, target):
    features, target = remove_contentless_images(features, target)
    features, target = apply__moving_average(window_size, features, target)
    features = features.T  # MinMaxScaler normalizes columns-wise only. But the flattened images are row-wise inside df
    features, target, scaler__target = normalize_data(features, target)  # normalize each image separately

    features = pd.DataFrame(features).T  # re-arrange image DataFrame to old order where images are sorted row-wise.
    features = transform_to_scans(features)
    # show_image(features[0])
    target = map(lambda x: x[0], target)  # turn target from Numpy Array back to Pandas Series
    target = pd.Series(target)

    feat__train, target__train, feat__test, target__test = split__train_and_test(features, target)
    return feat__train, target__train, feat__test, target__test, scaler__target


def preprocessing(window_size: object, features: object, target: object, target__LF: object = None) -> object:
    '''smooth graphs (moving average) + normalize input data + split HF into train and test data'''

    if target__LF is not None:
        return preprocessing__MultiFidelity(window_size, features, target, target__LF)
    else:
        return preprocessing__SingleFidelity(window_size, features, target)


def moving_average(window_size, Pandas_Series):
    '''apply a moving average to a given series of values. Goal: smoothing + denoising the series'''

    Pandas_Series = Pandas_Series.rolling(window=window_size, center=True).mean()
    return Pandas_Series.dropna()


def apply__moving_average__MultiFidelity(window_size: object, feat: object, target__HF: object,
                                         target__LF: object) -> object:
    '''
    apply moving average to the train, val and test dataset
    COMMENTS: decomment to plot effect of Moving Average
    '''

    target__HF = moving_average(window_size, target__HF)
    target__LF = moving_average(window_size, target__LF)

    common_idx = feat.index.intersection(target__HF.index)
    feat = feat.loc[common_idx]

    target__HF = target__HF.reset_index(drop=True)
    target__LF = target__LF.reset_index(drop=True)
    feat = feat.reset_index(drop=True)

    return feat, target__HF, target__LF


def apply__moving_average(window_size, feat, target):
    '''
    apply moving average to the train, val and test dataset
    COMMENTS: decomment to plot effect of Moving Average
    '''

    target = moving_average(window_size, target)

    common_idx = feat.index.intersection(target.index)
    feat = feat.loc[common_idx]

    target = target.reset_index(drop=True)
    feat = feat.reset_index(drop=True)

    return feat, target


def load_pkl(file):
    with open('train_test_dataset_from_troll.pkl', 'rb') as f:
        train_test_dataset_from_troll = pickle.load(f)

    return train_test_dataset_from_troll


def load_HFdataset(config, file):
    '''
    preprocess HFdataset.
    The each feature possesses multiple fidelities
    '''

    dataset = load_pkl(resource_path(file))

    # with open(file, 'rb') as f:
    #    train_test_dataset_from_troll = pickle.load(f)

    target__train__HF = dataset.dataset['scm'].train_test_dataset['train']['y'].T[0]
    feat__train__HF__table = dataset.dataset['scm'].train_test_dataset['train']['X']
    feat__train__HF__image = dataset.dataset['scm'].train_test_dataset['train']['surface']

    target__train__LF = dataset.dataset['troll'].train_test_dataset['train']['y'].T[0]
    feat__train__LF__table = dataset.dataset['troll'].train_test_dataset['train']['X']
    feat__train__LF__image = dataset.dataset['troll'].train_test_dataset['train']['surface']

    feat__train__HF__image = pd.DataFrame(feat__train__HF__image)
    target__train__HF = pd.Series(target__train__HF)
    target__train__LF = pd.Series(target__train__LF)

    (feat__train, target__train__HF, target__train__LF,
     feat__test, target__test__HF, target__test__LF,
     scaler__train_target) = preprocessing(config["window_size"]["test"], feat__train__HF__image, target__train__HF,
                                           target__train__LF)

    return feat__train, target__train__HF, target__train__LF, feat__test, target__test__HF, target__test__LF


def load_LFdataset(config, file):
    '''retrieve LF dataset for training AE and NNL. Use LF dataset as Source data'''

    train = pd.read_csv(resource_path(file), header=None)
    features = train.iloc[:, :4096]
    outputs = train.iloc[:, 4096:]
    target = outputs.iloc[:, 0]

    (train_features, train_target,
     feat__test, target__test,
     scaler__train_target) = preprocessing(config["window_size"]["train"], features, target)

    return train_features, train_target, feat__test, target__test


def load_TrainData(config, file):
    ''' load Training Data as DataFrame into program '''

    if file == config["file__train__HF"]:
        return load_HFdataset(config, file)

    else:
        return load_LFdataset(config, file)


def load_TestData(config, file, index):
    '''load Test Data as DataFrame into program'''

    test = pd.read_csv(resource_path(file), header=None)

    run_indx = test.iloc[:, -1] == index
    test_features = test.iloc[:, :4096][run_indx]
    test_outputs = test.iloc[:, 4096:][run_indx]
    test_target = test_outputs.iloc[:, 0]

    test_features, test_target, scaler__test_target = preprocessing(config["window_size"]["test"], test_features,
                                                                    test_target)

    return test_features, test_target


def load_CommonDataset(config):
    '''
    HF + LF features and targets
    '''

    train_features, train_target = load_TrainData(config["file__train__HF"])  # load whole Train Dataset
    test_features, test_target = load_TrainData(config["file__test"])  # load whole Test Dataset
    features__LF, target__LF = load_TrainData(config["file__train__LF"])

    features__HF = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)  # retrieve common Dataset
    target__HF = pd.concat([train_target, test_target], axis=0).reset_index(drop=True)

    return features__HF, target__HF, features__LF, target__LF


def normalize(df):
    '''
    Normalize Pandas object column-wise
    Each column gets normalized separately
    '''

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    return scaled, scaler


def de_normalize(df, scaler):
    unscaled = scaler.inverse_transform(df)
    return unscaled


def normalize_data__MultiFidelity(feat, target__HF, target__LF):
    '''normalize a feature and all its targets (several fidelities)'''

    feat, _ = normalize(feat)
    target__HF, scaler__target__HF = normalize(pd.DataFrame(target__HF))
    target__LF, scaler__target__LF = normalize(pd.DataFrame(target__LF))

    return feat, target__HF, target__LF, scaler__target__HF


def normalize_data(feat: object, target: object) -> object:
    feat, _ = normalize(feat)
    target, scaler__target = normalize(pd.DataFrame(target))

    return feat, target, scaler__target


def transform_to_scans(x_data):
    surface_scans = []
    for i in range(x_data.shape[0]):
        surface_scans.append(np.reshape(np.array(x_data.iloc[i, :]), (64, 64, 1)))
    surface_scans = np.array(surface_scans)

    return surface_scans
