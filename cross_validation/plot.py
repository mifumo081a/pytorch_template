from ..pickle_io import *
import os
from pytorch_template.utils import get_dataloaders
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob


def plot_kfold_dist(kfold_root, logs_root=os.getcwd(), save=False):
    save_root = os.path.join(logs_root, "kfold_dist")
    os.makedirs(save_root, exist_ok=True)
    for path in glob(os.path.join(kfold_root, "*/")):
        fold_num = int(path.split("/")[-2])

        dataloaders = get_dataloaders(path)
        labels = dataloaders["train"].dataset.dataset.labels
        labels_list = []
        trainval_list = []

        for phase in ["train", "val"]:
            for _, targets in dataloaders[phase]:
                if targets.shape:
                    trainval_list.extend([phase for i in range(len(targets))])
                    labels_list.extend([labels[target] for target in targets.tolist()])
                else:
                    trainval_list.append(phase)
                    labels_list.append(labels[targets.item()])
        
        df = pd.DataFrame({"trainval": trainval_list, "label": labels_list})
                    
        plt.figure()
        sns.histplot(data=df, x="label", hue="trainval", multiple="dodge", shrink=.8)
        plt.xticks(rotation=45)
        if save:
            plt.savefig(os.path.join(save_root, str(fold_num)+".png"))
        else:
            plt.show()