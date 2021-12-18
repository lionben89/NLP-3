import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay

def autolabel(rects, labels, ax):
    for idx,rect in enumerate(rects):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                labels[idx],
                ha='center', va='bottom', rotation=0)

def plot_bars(classifiers,scores, metric):
    fig, ax = plt.subplots(layout='constrained')
    ax.set_title(metric)
    ax.set_ylabel("score")
    bar_plot = ax.bar(classifiers, scores);
    autolabel(bar_plot,scores,ax)
    fig.savefig("{}.png".format(metric))
    
def plot_loss_graphs(classifiers,losses):
    fig, ax = plt.subplots()
    x = np.linspace(0,1,len(losses[0]))
    for i in range(len(classifiers)):
        ax.plot(x, losses[i], label=classifiers[i])
    ax.set_title("Train Loss")
    ax.legend()
    fig.savefig("train_loss.png")
    
def plot_rocs(classifiers, rocs):
    fig, ax = plt.subplots()
    for i in range(len(classifiers)):
        ax.plot(rocs['fpr'][i], rocs['tpr'][i], label=classifiers[i])
    ax.set_title("Roc curves")
    ax.set_xlabel("Fpr")
    ax.set_ylabel("Tpr")
    ax.legend()
    fig.savefig("roc_curves.png")
    
def plot_all(data):
    losses = []
    losses_classifiers = []
    classifiers = []
    bars = {'accuracy':[],'precision':[],'recall':[],'auc':[],'f1':[]}
    rocs = {'fpr':[],'tpr':[]}
    
    ## collect data
    for c_data in data:
        cls_name = "{}_{}".format(c_data["classifier"].to_string(),c_data["vectorize"].to_string())
        classifiers.append(cls_name)
        if (hasattr(c_data["classifier"], 'losses')):
            losses.append(c_data["classifier"].losses)
            losses_classifiers.append(cls_name)
        for metric in bars.keys():
            bars[metric].append(c_data["scores"][metric])
        rocs['fpr'].append(c_data["fpr"])
        rocs['tpr'].append(c_data["tpr"])
            
    ## plot bars data
    for metric in bars.keys():
        plot_bars(classifiers,bars[metric],metric)
    
    ## plot losses
    if (len(losses)>0):
        plot_loss_graphs(losses_classifiers,losses)
    
    ## plot rocs
    plot_rocs(classifiers,rocs)
