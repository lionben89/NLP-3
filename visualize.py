import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay

def plot_bars(classifiers,scores, metric):
    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    ax.set_title(metric)
    ax.set_ylabel("score")
    ax.bar(classifiers, scores);
    fig.savefig("{}.png".format(metric))
    
def plot_loss_graphs(classifiers,losses):
    fig, ax = plt.subplots(figsize=(5, 2.7))
    x = np.linspace(0,1,len(losses[0]))
    for i in range(len(classifiers)):
        ax.plot(x, losses[i], label=classifiers[i])
    ax.set_title("Train Loss")
    ax.legend()
    fig.savefig("train_loss.png")
    
def plot_rocs(classifiers, rocs):
    fig, ax = plt.subplots(figsize=(5, 2.7))
    for i in range(len(classifiers)):
        ax.plot(rocs['fpr'][i], rocs['tpr'][i], label=classifiers[i])
    ax.set_title("Roc curves")
    ax.set_xlabel("Fpr")
    ax.set_ylabel("Tpr")
    ax.legend()
    fig.savefig("roc_curves.png")
    
def plot_all(data):
    losses = []
    classifiers = []
    bars = {'accuracy':[],'precision':[],'recall':[],'auc':[],'f1':[]}
    rocs = {'fpr':[],'tpr':[]}
    
    ## collect data
    for c_data in data:
        classifiers.append("{}_{}".format(c_data["classifier"].to_string(),c_data["vectorize"].to_string()))
        losses.append(c_data["classifier"].losses)
        for metric in bars.keys():
            bars[metric].append(c_data["scores"][metric])
        rocs['fpr'].append(c_data["vectorize"]["fpr"])
        rocs['tpr'].append(c_data["vectorize"]["tpr"])
            
    ## plot bars data
    for metric in bars.keys():
        plot_bars(classifiers,bars[metric],metric)
    
    ## plot losses
    plot_loss_graphs(classifiers,losses)
    
    ## plot rocs
    plot_rocs(classifiers,rocs)
