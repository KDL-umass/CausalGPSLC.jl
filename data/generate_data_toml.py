import toml
import numpy as np


def generate_dict(noises, assignments, parameters, udim=1, xdim=1, xvar=1.0, uconv=1.0, ttype="continuous", aggop="+",
                  data_size=100, obj_size=10, eps=1e-13):
    data = dict()
    data_dict = dict()
    # total number of data
    data_dict['n'] = data_size

    # sigma U
    data_dict['obj_size'] = obj_size
    data_dict['eps'] = eps
    data_dict['ucov'] = uconv
    data_dict['udim'] = udim

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
    data_dict['UaggOp'] = aggop

    data_dict['Ttype'] = ttype

    # functional representation
    data_dict['XTAssignment'] = assignments['XT']
    data_dict['UTAssignment'] = assignments['UT']
    data_dict['XYAssignment'] = assignments['XY']
    data_dict['TYAssignment'] = assignments['TY']
    data_dict['UYAssignment'] = assignments['UY']
    data_dict['YUAssignment'] = assignments['YU']
    data_dict['TUAssignment'] = assignments['TU']

    # put everything together
    data['data'] = data_dict
    data_dict['XTparams'] = parameters['XT']
    data_dict['UTparams'] = parameters['UT']
    data_dict['XYparams'] = parameters['XY']
    data_dict['TYparams'] = parameters['TY']
    data_dict['UYparams'] = parameters['UY']
    data_dict['YUparams'] = parameters['YU']
    data_dict['TUparams'] = parameters['TU']

    return data


def XT_params(mechanisms, operation):
    """
    :param mechanisms: {["linear"], ["polynomial"], ["polynomial", "sinusoidal"]}
    :param operation: {"+", "*"}
    :return:
    """
    out = dict()
    if len(mechanisms) > 1:
        op = "*"
    else:
        op = "+"
    out["aggOp"] = op
    if "linear" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
        else:
            out["poly"] = [1.0, 1.0]
    elif "sinusoidal" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
            out["sin"] = [0.0, 1.0, 1.0]
        else:
            out["poly"] = [1.0, 0.0]
            out["sin"] = [0.0, 1.0, 1.0]
    else:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
        else:
            out["poly"] = [-1.0, 1.0]
    return out


def UT_params(mechanisms, operation):
    """
    :param mechanisms: {["linear"], ["polynomial"], ["polynomial", "sinusoidal"]}
    :param operation: {"+", "*"}
    :return:
    """
    out = dict()
    if len(mechanisms) > 1:
        op = "*"
    else:
        op = "+"
    out["aggOp"] = op
    if "linear" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
        else:
            out["poly"] = [0.0, 1.0]
    elif "sinusoidal" in mechanisms:
        if operation == '+':
            out["poly"] = [1.0, 0.0]
            out["sin"] = [0.0, 1.0, 1.5]
        else:
            out["poly"] = [1.0, 0.0]
            out["sin"] = [0.0, 1.0, 1.5]
    else:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
        else:
            out["poly"] = [0.0, 1.0]
    return out


def XY_params(mechanisms, operation):
    """
    :param mechanisms: {["linear"], ["polynomial"], ["polynomial", "sinusoidal"]}
    :param operation: {"+", "*"}
    :return:
    """
    out = dict()
    if len(mechanisms) > 1:
        op = "*"
    else:
        op = "+"
    out["aggOp"] = op
    if "linear" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
        else:
            out["poly"] = [1.0, 1.0]
    elif "sinusoidal" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
            out["sin"] = [0.0, 1.0, 1.0]
        else:
            out["poly"] = [1.0, 0.0]
            out["sin"] = [0.0, 1.0, 1.0]
    else:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
        else:
            out["poly"] = [1.0, 1.0]
    return out


def TY_params(mechanisms, operation):
    """
    :param mechanisms: {["linear"], ["polynomial"], ["polynomial", "sinusoidal"]}
    :param operation: {"+", "*"}
    :return:
    """
    out = dict()
    if len(mechanisms) > 1:
        op = "*"
    else:
        op = "+"
    out["aggOp"] = op
    if "linear" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
        else:
            out["poly"] = [1.0, 1.0]
    elif "sinusoidal" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 3.0]
            out["sin"] = [0.0, 1.0, 2.0]
        else:
            out["poly"] = [0.5, 0.0]
            out["sin"] = [0.0, 1.0, 2.0]
    else:
        if operation == '+':
            out["poly"] = [0.0, 0.0, 0.0, 1.0]
        else:
            out["poly"] = [1.0, 0.0, 1.0]
    return out


def UY_params(mechanisms, operation):
    """
    :param mechanisms: {["linear"], ["polynomial"], ["polynomial", "sinusoidal"]}
    :param operation: {"+", "*"}
    :return:
    """
    out = dict()
    if len(mechanisms) > 1:
        op = "*"
    else:
        op = "+"
    out["aggOp"] = op
    if "linear" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 3.0]
        else:
            out["poly"] = [0.0, 1.0]
    elif "sinusoidal" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 2.0]
            out["sin"] = [0.0, 1.0, 3.0]
        else:
            out["poly"] = [0.5, 0.0]
            out["sin"] = [0.0, 1.0, 2.0]
    else:
        if operation == '+':
            out["poly"] = [0.0, 5.0, 1.0]
        else:
            out["poly"] = [0.0, 1.0]
    return out


def TU_params(mechanisms, operation):
    """
    :param mechanisms: {["linear"], ["polynomial"], ["polynomial", "sinusoidal"]}
    :param operation: {"+", "*"}
    :return:
    """
    out = dict()
    if len(mechanisms) > 1:
        op = "*"
    else:
        op = "+"
    out["aggOp"] = op
    if "linear" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
        else:
            out["poly"] = [0.0, 1.0]
    elif "sinusoidal" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
            out["sin"] = [0.0, 0.0, 3.0]
        else:
            out["poly"] = [0.0, 1.0]
            out["sin"] = [0.0, 0.0, 2.0]
    else:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
        else:
            out["poly"] = [0.0, 1.0]
    return out


def YU_params(mechanisms, operation):
    """
    :param mechanisms: {["linear"], ["polynomial"], ["polynomial", "sinusoidal"]}
    :param operation: {"+", "*"}
    :return:
    """
    out = dict()
    if len(mechanisms) > 1:
        op = "*"
    else:
        op = "+"
    out["aggOp"] = op
    if "linear" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
        else:
            out["poly"] = [0.0, 1.0]
    elif "sinusoidal" in mechanisms:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
            out["sin"] = [0.0, 0.0, 3.0]
        else:
            out["poly"] = [0.0, 1.0]
            out["sin"] = [0.0, 0.0, 2.0]
    else:
        if operation == '+':
            out["poly"] = [0.0, 1.0]
        else:
            out["poly"] = [0.0, 1.0]
    return out


def generate_toml(fname, raw_data):
    new_toml_string = toml.dumps(raw_data)
    with open(fname, 'w') as f:
        f.write(new_toml_string)


def main():
    function = (["linear"], ["polynomial"], ["polynomial", "sinusoidal"])
    interactions = ["+", "x"]
    Ttype = ["continuous", "binary"]
    # xdim = [0, 1, 3, 5, 10, 30, 50, 100, 300, 500]
    # obj_dep = np.arange(11)/10
    # noise_level = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 1e1, 5e1, 1e2]
    count = 1
    for m in function:
        for i in interactions:
            for tt in Ttype:
                noises = [0.5, 0.5, 0.5, 0.5]
                params = dict()
                params['XT'] = XT_params(m, i)
                params['UT'] = UT_params(m, i)
                params['XY'] = XY_params(m, i)
                params['UY'] = UY_params(m, i)
                params['TY'] = TY_params(m, i)
                params['TU'] = TU_params(m, i)
                params['YU'] = YU_params(m, i)

                interaction = ('multi' if i == 'x' else 'additive')

                name = np.copy(m).tolist()
                if "linear" in m:
                    mech = ["polynomial"]
                else:
                    mech = m

                name.append(interaction)
                name.append(tt)
                functions = dict()
                functions['XT'] = mech
                functions['UT'] = mech
                functions['XY'] = mech
                functions['UY'] = mech
                functions['TY'] = mech
                functions['TU'] = mech
                functions['YU'] = mech

                raw_string = generate_dict(noises, functions, params, data_size=200, obj_size=10, uconv=1.0,
                                           aggop=i, ttype=tt, xdim=3, udim=3)
                generate_toml("./synthetic/"+str(count)+'.toml', raw_string)
                count += 1


if __name__ == '__main__':
    main()
