import matplotlib
import numpy as np
from prettytable import PrettyTable


def Statistical(val):
    out = np.zeros((5))
    out[0] = max(val)
    out[1] = min(val)
    out[2] = np.mean(val)
    out[3] = np.median(val)
    out[4] = np.std(val)
    return out


def Plot_seg_Results():
    matplotlib.use('TkAgg')
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    eval = np.load('Eval_seg.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Dice', 'Jaccard']
    Algorithm = ['RSA-ADSL', 'TFMOA-ADSL', 'SCO-ADSL', 'EVO-ADSL', 'RPU-EVO-ADSL']
    value = eval[:, :, :]
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value[4, j, :])
    print(
        '--------------------------------------------------Segmentation--------------------------------------------------')
    print(Table)
    stat = np.zeros((value.shape[1], value.shape[2], 5))
    Value = np.zeros((eval.shape[0], 5))
    for j in range(eval.shape[0]):
        Value[j, 0] = np.min(eval[j, :])
        Value[j, 1] = np.max(eval[j, :])
        Value[j, 2] = np.mean(eval[j, :])
        Value[j, 3] = np.median(eval[j, :])
        Value[j, 4] = np.std(eval[j, :])


if __name__ == '__main__':
    Plot_seg_Results()
