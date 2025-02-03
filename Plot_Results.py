import warnings
from itertools import cycle
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")
from Global_Vars import Global_Vars


def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i - 0.20, np.round(y[i], 3) / 2, str(np.round(y[i], 2)) + '%')


def plot_results_conv():
    conv = np.load('Fitness.npy', allow_pickle=True)


    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['RSA-ADSL', 'TFMOA-ADSL', 'SCO-ADSL', 'EVO-ADSL', 'RPU-EVO-ADSL']

    Value = np.zeros((conv.shape[0], 5))
    for j in range(conv.shape[0]):
        Value[j, 0] = np.min(conv[j, :])
        Value[j, 1] = np.max(conv[j, :])
        Value[j, 2] = np.mean(conv[j, :])
        Value[j, 3] = np.median(conv[j, :])
        Value[j, 4] = np.std(conv[j, :])

    Table = PrettyTable()
    Table.add_column("ALGORITHMS", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print(
        '-------------------------------------------------- Statistical Analysis--------------------------------------------------')
    print(Table)

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    iteration = np.arange(conv.shape[1])
    plt.plot(iteration, conv[0, :], color='m', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
             label='RSA-ADSL')
    plt.plot(iteration, conv[1, :], color='c', linewidth=3, marker='p', markerfacecolor='green', markersize=12,
             label='TFMOA-ADSL')
    plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='.', markerfacecolor='cyan', markersize=12,
             label='SCO-ADSL')
    plt.plot(iteration, conv[3, :], color='r', linewidth=3, marker='o', markerfacecolor='magenta', markersize=12,
             label='EVO-ADSL')
    plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black', markersize=12,
             label='RPU-EVO-ADSL')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    path = "./Results/Conv.png"
    plt.savefig(path)
    plt.show()


def Plot_ROC():
    lw = 2
    cls = ['GoogleNet', 'AlexNet', 'HDCNet', 'ADSL', 'RPU-EVO-ADSL']
    colors1 = cycle(["#65fe08", "#4e0550", "#f70ffa", "#a8a495", "#004577", ])
    fig = plt.figure()
    fig.canvas.manager.set_window_title('ROC')
    for n in range(1):
        for i, color in zip(range(5), colors1):  # For all classifiers
            Predicted = np.load('roc_score.npy', allow_pickle=True)[n][i].astype('float')
            Actual = np.load('roc_act.npy', allow_pickle=True)[n][i].astype('int')
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label="{0}".format(cls[i]),
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/roc.png"
        plt.savefig(path)
        plt.show()


def Confusion_matrix():
    Eval = np.load('Eval_all_Batchsize.npy', allow_pickle=True)
    # value1 = eval[4, 4, :5]
    value = Eval[3, 4, :5]
    val = np.asarray([0, 1, 1])
    data = {'y_Actual': [val.ravel()],
            'y_Predicted': [np.asarray(val).ravel()]
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'],
                                   colnames=['Predicted'])
    value = value.astype('int')

    confusion_matrix.values[0, 0] = value[1]
    confusion_matrix.values[0, 1] = value[3]
    confusion_matrix.values[1, 0] = value[2]
    confusion_matrix.values[1, 1] = value[0]
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Confusion Matrix')
    sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = ' + str(Eval[3, 4, 4] * 100)[:5] + '%')
    sn.plotting_context()
    path = './Results/Confusion.png'
    plt.savefig(path)
    plt.show()


def plot_results():
    eval = np.load('Eval_all_Batchsize.npy', allow_pickle=True)
    eval1 = eval
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Algorithm = ['TERMS', 'RSA-ADSL', 'TFMOA-ADSL', 'SCO-ADSL', 'EVO-ADSL', 'RPU-EVO-ADSL']
    Classifier = ['TERMS', 'Googlenet', 'Alexnet', 'HDCnet', 'ADSL', 'RPU-EVO-ADSL']
    value1 = eval[3, :, 4:]

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value1[j, :])
    print('-------------------------------------------------- Batch_Size - Algorithm Comparison ',
          '--------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
    print('-------------------------------------------------- Batch_Size - Classifier Comparison',
          '--------------------------------------------------')
    print(Table)
    Batch_Size = [1, 2, 3, 4]
    for j in range(len(Graph_Terms)):
        Graph = np.zeros(eval.shape[0:2])
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                if j == 9:
                    Graph[k, l] = eval[k, l, Graph_Terms[j] + 4]
                else:
                    Graph[k, l] = eval[k, l, Graph_Terms[j] + 4]
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Batch size - Algorithm')
        plt.plot(Batch_Size, Graph[:, 0], color='r', linewidth=3, marker='*', markerfacecolor='#8B6969',
                 markersize=16,
                 label="RSA-ADSL")
        plt.plot(Batch_Size, Graph[:, 1], color='#9A32CD', linewidth=3, marker='8', markerfacecolor='#FF4500',
                 markersize=12,
                 label="TFMOA-ADSL")
        plt.plot(Batch_Size, Graph[:, 2], color='#FF1493', linewidth=3, marker='H', markerfacecolor='cyan',
                 markersize=12,
                 label="SCO-ADSL")
        plt.plot(Batch_Size, Graph[:, 3], color='b', linewidth=3, marker='8', markerfacecolor='#CD2990',
                 markersize=12,
                 label="EVO-ADSL")
        plt.plot(Batch_Size, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='black',
                 markersize=12,
                 label="RPU-EVO-ADSL")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)

        plt.xticks(Batch_Size, ('4', '16', '32', '128'))
        plt.xlabel('Batch_Size')
        plt.ylabel(Terms[Graph_Terms[j]])
        path = "./Results/_%s_line.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()

        fig = plt.figure()
        fig.canvas.manager.set_window_title('Batch Size - Method')
        ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
        X = np.arange(4)
        ax.bar(X + 0.00, Graph[:, 5], color='#FF9912', width=0.10, label="Googlenet")
        ax.bar(X + 0.10, Graph[:, 6], color='#00EEEE', width=0.10, label="Alexnet")
        ax.bar(X + 0.20, Graph[:, 7], color='#D15FEE', width=0.10, label="HDCnet")
        ax.bar(X + 0.30, Graph[:, 8], color='#FFAEB9', width=0.10, label="ADSL")
        ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="RPU-EVO-ADSL")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                   ncol=3, fancybox=True, shadow=True)
        plt.xticks(X + 0.20, ('4', '16', '32', '128'))
        plt.xlabel('Batch_Size')
        plt.ylabel(Terms[Graph_Terms[j]])
        path = "./Results/%s_bar.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()


if __name__ == '__main__':
    plot_results_conv()
    Confusion_matrix()
    Plot_ROC()
    plot_results()
