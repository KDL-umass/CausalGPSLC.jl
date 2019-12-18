import toml
import numpy as np


def generate_dict(noises, assignments, parameters, xdim=1, xvar=1.0, uconv=1.0, ttype="continuous", aggop="+",
                  data_size=10000, obj_size=100, eps=1e-13):
    data = dict()
    data_dict = dict()
    # total number of data
    data_dict['n'] = data_size

    # sigma U
    data_dict['obj_size'] = obj_size
    data_dict['eps'] = eps
    data_dict['ucov'] = uconv

    # sigma X
    data_dict['xvar'] = xvar
    data_dict['xdim'] = xdim

    # noises
    data_dict['uNoise'] = noises[0]
    data_dict['xNoise'] = noises[1]
    data_dict['tNoise'] = noises[2]
    data_dict['yNoise'] = noises[3]

    # aggregation operation
    data_dict['TaggOp'] = aggop
    data_dict['YaggOp'] = aggop

    data_dict['Ttype'] = ttype

    # functional representation
    data_dict['XTAssignment'] = assignments
    data_dict['UTAssignment'] = assignments
    data_dict['XYAssignment'] = assignments
    data_dict['TYAssignment'] = assignments
    data_dict['UYAssignment'] = assignments

    # put everything together
    data['data'] = data_dict
    data_dict['XTparams'] = parameters
    data_dict['UTparams'] = parameters
    data_dict['XYparams'] = parameters
    data_dict['TYparams'] = parameters
    data_dict['UYparams'] = parameters

    return data


def default_params(mechanisms, t, tt):
    out = dict()
    if len(mechanisms)>1:
        op = "*"
    else:
        op = "+"
    out["aggOp"] = op

    if "polynomial" in mechanisms and len(mechanisms) > 1:
        out["poly"] = [1.0, 1.0]
        if tt != 'binary':
            out["poly"][0] = 0.0

    elif "polynomial" in mechanisms:
        out["poly"] = [0.1, 0.0, 1.0, 1.0]
        if t == "x":
            out["poly"] = [0.1, 0.0, 0.0, 0.1]
        if tt != 'binary':
            out["poly"][0] = 0.0

    if "linear" in mechanisms:
        out["poly"] = [1.0, 1.0]
        if tt != 'binary':
            out["poly"][0] = 0.0

    if "sinusoidal" in mechanisms:
        out["sin"] = [0.0, 1.0, 1.0]
    return out


def generate_toml(fname, raw_data):
    new_toml_string = toml.dumps(raw_data)
    with open(fname, 'w') as f:
        f.write(new_toml_string)


def main():
    mechanisms = (["linear"], ["polynomial"], ["polynomial", "sinusoidal"])
    interactions = ["+", "x"]
    Ttype = ["continuous", "binary"]
    # xdim = [0, 1, 3, 5, 10, 30, 50, 100, 300, 500]
    # obj_dep = np.arange(11)/10
    # noise_level = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 1e1, 5e1, 1e2]
    for m in mechanisms:
        for i in interactions:
            for tt in Ttype:
                noises = [1.0, 1.0, 1.0, 1.0]
                params = default_params(m, i, tt)
                interaction = ('multi' if i == 'x' else 'additive')

                name = np.copy(m).tolist()
                if "linear" in m:
                    mech = ["polynomial"]
                else:
                    mech = m

                name.append(interaction)
                name.append(tt)
                raw_string = generate_dict(noises, mech, params, aggop=i, ttype=tt)

                fname = '_'.join(name)
                fname = './synthetic/' + fname + '.toml'
                generate_toml(fname, raw_string)


if __name__ == '__main__':
    main()
