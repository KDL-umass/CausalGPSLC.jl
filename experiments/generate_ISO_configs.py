import toml
import numpy as np
import os

def generate_dict(model_type, bias, posterior_dir, data_dir="../data/ISO/2018_smd_hourly.csv", 
                    mean=45, new_var=15, n_samples_per_state=25, shape=4.0, scale=4.0, nU=3, 
                    nOuter=5000, nMHInner=3, nESInner=5):
    data = dict()
    paths = dict()
    downsampling = dict()
    model_hyperparameters = dict()
    model = dict()
    inference = dict()
    
    # Paths
    paths['data'] = data_dir
    paths['posterior_dir'] = posterior_dir

    # Downsampling
    downsampling["bias"] = bias
    downsampling["mean"] = mean
    downsampling["newVar"] = new_var
    downsampling["nSamplesPerState"] = n_samples_per_state

    # model_hyperparameters
    model_hyperparameters['uNoiseShape'] = shape
    model_hyperparameters['uNoiseScale'] = scale
    model_hyperparameters['xNoiseShape'] = shape
    model_hyperparameters['xNoiseScale'] = scale
    model_hyperparameters['tNoiseShape'] = shape
    model_hyperparameters['tNoiseScale'] = scale
    model_hyperparameters['yNoiseShape'] = shape
    model_hyperparameters['yNoiseScale'] = scale

    model_hyperparameters['uxLSShape'] = shape
    model_hyperparameters['uxLSScale'] = scale
    model_hyperparameters['utLSShape'] = shape
    model_hyperparameters['utLSScale'] = scale
    model_hyperparameters['xtLSShape'] = shape
    model_hyperparameters['xtLSScale'] = scale
    model_hyperparameters['uyLSShape'] = shape
    model_hyperparameters['uyLSScale'] = scale
    model_hyperparameters['xyLSShape'] = shape
    model_hyperparameters['xyLSScale'] = scale
    model_hyperparameters['tyLSShape'] = shape
    model_hyperparameters['tyLSScale'] = scale

    model_hyperparameters['xScaleShape'] = shape
    model_hyperparameters['xScaleScale'] = scale
    model_hyperparameters['tScaleShape'] = shape
    model_hyperparameters['tScaleScale'] = scale
    model_hyperparameters['yScaleShape'] = shape
    model_hyperparameters['yScaleScale'] = scale

    # model
    model['type'] = model_type
    model['nU'] = nU

    # inference
    inference['nOuter'] = nOuter
    inference['nMHInner'] = nMHInner
    inference['nESInner'] = nESInner

    # put everything together
    data['downsampling'] = downsampling
    data['paths'] = paths
    data['model_hyperparameters'] = model_hyperparameters
    data['model'] = model
    data['inference'] = inference

    return data

def generate_toml(fname, raw_data):
    new_toml_string = toml.dumps(raw_data)
    with open(fname, 'w') as f:
        f.write(new_toml_string)

def main():
    biases = [i for i in range(10)]
    nSamplesPerStates = [2, 5, 10, 15, 20, 25]

    model_types = ["correct", "no_confounding", "no_objects", "GP_per_object"]
    i = 1
    for bias in biases:
        for model_type in model_types:
            posterior_dir = "results/ISO/bias{}/{}/".format(bias, model_type)
            raw_string = generate_dict(model_type, bias, posterior_dir)
            generate_toml("./config/ISO/{}.toml".format(i), raw_string)
            os.makedirs(posterior_dir, exist_ok=True)
            i += 1

    for nSamplesPerState in nSamplesPerStates:
        for model_type in model_types:
            posterior_dir = "results/ISO/nSamplesPerState{}/{}/".format(nSamplesPerState, model_type)
            raw_string = generate_dict(model_type, 9, posterior_dir, n_samples_per_state=nSamplesPerState)
            generate_toml("./config/ISO/{}.toml".format(i), raw_string)
            os.makedirs(posterior_dir, exist_ok=True)
            i += 1


if __name__ == "__main__":
    main()