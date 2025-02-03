import numpy as np
from Evaluation_All import seg_evaluation
from Global_Vars import Global_Vars
from Model_ADSL_New import Model_ADSL
from Model_Trans_Unet import Model_Trans_Unet
from Relief_Score import relief_score


def Objective_Seg(Soln):
    Images = Global_Vars.Images
    Target = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            pred = Model_Trans_Unet(Images, Target, sol)
            Eval = seg_evaluation(pred, Target)
            Fitn[i] = 1 / Eval[5]
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        pred = Model_Trans_Unet(Images, Target, sol)
        Eval = seg_evaluation(pred, Target)
        Fitn = 1 / Eval[5]
        return Fitn


def Objfun_Feat(Soln):
    Feat = Global_Vars.Images
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        Selected_Feature = Feat[:, np.round(sol).astype('int')]
        Fitn[i] = 1 / relief_score(Selected_Feature)
    return Fitn


def Objfun_Cls(Soln):
    image = Global_Vars.Images
    target = Global_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        learnper = round(image.shape[0] * 0.75)
        train_data = image[learnper:, :]
        train_target = target[learnper:, :]
        test_data = image[:learnper, :]
        test_target = target[:learnper, :]
        Eval = Model_ADSL(train_data, train_target, test_data, test_target, sol.astype('int'))
        Fitn[i] = (1 / Eval[4]) + Eval[11]
    return Fitn
