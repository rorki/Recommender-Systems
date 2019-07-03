import os

############ create all dirs required to save model ###########
def make_out_dirs(model_name, xp_name):
    # general path for this file
    XP_HOME = "D:/Models/thesis/" + model_name
    # name of current experiemnt
    XP_NAME = xp_name

    XP_PATH = '%s/%s/' % (XP_HOME, XP_NAME)
    U_V_PATH = '%s/%s/pickles/' % (XP_HOME, XP_NAME)
    MODEL_PATH = '%s/%s/tf/' % (XP_HOME, XP_NAME)

    if not os.path.isdir(XP_HOME):
        os.mkdir(XP_HOME)

    if not os.path.isdir(XP_PATH):
        os.mkdir(XP_PATH)

    if not os.path.isdir(U_V_PATH):
        os.mkdir(U_V_PATH)

    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)
        
    return XP_PATH, U_V_PATH, MODEL_PATH