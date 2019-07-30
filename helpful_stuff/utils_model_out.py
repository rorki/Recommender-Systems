import os

ROOT = "D:/Models/thesis/"


def make_out_dirs(model_name, xp_name):
    """
    Creates structure of output directories for given model and experiment.

    :param model_name: str, name of model
    :param xp_name: str, name of experiment
    :return: path to experiment folder, to user and item matrices and to model parameters
    """
    # root path for model
    xp_home = ROOT + model_name

    xp_path = '%s/%s/' % (xp_home, xp_name)
    mx_path = '%s/%s/pickles/' % (xp_home, xp_name)
    model_path = '%s/%s/tf/' % (xp_home, xp_name)

    if not os.path.isdir(xp_home):
        os.mkdir(xp_home)

    if not os.path.isdir(xp_path):
        os.mkdir(xp_path)

    if not os.path.isdir(mx_path):
        os.mkdir(mx_path)

    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        
    return xp_path, mx_path, model_path


if __name__ == "__main__":
    make_out_dirs('test', 'test')
    assert os.path.exists(ROOT + 'test'), 'missed root path'
    assert os.path.exists(ROOT + 'test/test/pickles'), 'missed matrices path'
    assert os.path.exists(ROOT + 'test/test/tf'), 'missed model path'
    os.remove(ROOT + 'test')
