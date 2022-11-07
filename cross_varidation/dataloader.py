from typing import Dict, Optional, Any
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, GroupKFold
from torch.utils.data import Dataset, Subset, DataLoader
from ..pickle_io import *
import os
from pytorch_template.utils import get_dataloaders
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
import copy


def kfold_split(dataset: Dataset, test_dataset: Optional[Dataset]=None, transforms: Optional[Dict]=None, n_splits=5, train_size=0.7,
                root: str=os.getcwd(), groups:Any=[],
                num_workers=2, batch_size=16,
                shuffle=False, random_state=None, folder_name="kfold_pickle"):
    """
    transforms = {"train": , "val": , "test": }
    """
    kfold_root = os.path.join(root, folder_name)
    os.makedirs(kfold_root, exist_ok=True)
    
    if test_dataset is None:
        if len(groups) > 0:
            kf = GroupKFold(n_splits=n_splits)
            dataset_idx = np.array(range(len(dataset)))
            mapping = np.arange(groups.max()+1)
            if shuffle:
                rng = np.random.RandomState(random_state)
                rng.shuffle(mapping)

            for i, (trainval_idx, test_idx) in enumerate(kf.split(X=dataset_idx, groups=mapping[groups])):
                trainval_groups = groups[trainval_idx]
                gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
                train_inner_idx, val_inner_idx = next(gss.split(trainval_idx, groups=trainval_groups))
                train_idx, val_idx = trainval_idx[train_inner_idx], trainval_idx[val_inner_idx]

                print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
                train_dataset = Subset(copy.deepcopy(dataset), train_idx)
                val_dataset = Subset(copy.deepcopy(dataset), val_idx)
                test_dataset = Subset(copy.deepcopy(dataset), test_idx)
                
                if transforms is not None:
                    train_dataset.dataset.transform = transforms["train"]
                    val_dataset.dataset.transform = transforms["val"]
                    test_dataset.dataset.transform = transforms["test"]
                
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
                randomtest_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
                
                dataloaders  = {"train":train_dataloader, "val":val_dataloader}
                testloaders = {"test":test_dataloader, "randomtest":randomtest_dataloader}
                
                os.makedirs(os.path.join(kfold_root, str(i)),exist_ok=True)

                save_as_pickle(dataloaders,'dataloaders',os.path.join(kfold_root, str(i)))
                save_as_pickle(testloaders, "testloaders", os.path.join(kfold_root, str(i)))
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            dataset_idx = list(range(len(dataset)))
            for i, (trainval_idx, test_idx) in enumerate(kf.split(dataset_idx)):
                train_inner_idx, val_inner_idx = train_test_split(trainval_idx, train_size=train_size,
                                                            shuffle=shuffle, random_state=random_state)
                train_idx, val_idx = trainval_idx[train_inner_idx], trainval_idx[val_inner_idx]

                print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
                train_dataset = Subset(copy.deepcopy(dataset), train_idx)
                val_dataset = Subset(copy.deepcopy(dataset), val_idx)
                test_dataset = Subset(copy.deepcopy(dataset), test_idx)
                
                if transforms is not None:
                    train_dataset.dataset.transform = transforms["train"]
                    val_dataset.dataset.transform = transforms["val"]
                    test_dataset.dataset.transform = transforms["test"]
                
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
                randomtest_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
                
                dataloaders  = {"train":train_dataloader, "val":val_dataloader}
                testloaders = {"test":test_dataloader, "randomtest":randomtest_dataloader}
                
                os.makedirs(os.path.join(kfold_root, str(i)),exist_ok=True)

                save_as_pickle(dataloaders,'dataloaders',os.path.join(kfold_root, str(i)))
                save_as_pickle(testloaders, "testloaders", os.path.join(kfold_root, str(i)))
        
        print(f"All: {len(dataset)}")
        # dataset.transform = transforms["test"]
        # all_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        # save_as_pickle(all_dataloader, "allloader", kfold_root)
    
    else:
        if len(groups) > 0:
            kf = GroupKFold(n_splits=n_splits)
            dataset_idx = np.array(range(len(dataset)))
            mapping = np.arange(groups.max()+1)
            if shuffle:
                rng = np.random.RandomState(random_state)
                rng.shuffle(mapping)

            for i, (trainval_idx, val_idx) in enumerate(kf.split(X=dataset_idx, groups=mapping[groups])):
                print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
                train_dataset = Subset(copy.deepcopy(dataset), train_idx)
                val_dataset = Subset(copy.deepcopy(dataset), val_idx)
                
                if transforms is not None:
                    train_dataset.dataset.transform = transforms["train"]
                    val_dataset.dataset.transform = transforms["val"]
                
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                
                dataloaders  = {"train":train_dataloader, "val":val_dataloader}
                
                os.makedirs(os.path.join(kfold_root, str(i)),exist_ok=True)

                save_as_pickle(dataloaders,'dataloaders',os.path.join(kfold_root, str(i)))
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            for i, (train_idx, val_idx) in enumerate(kf.split(list(range(len(dataset))))):
                print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
                train_dataset = Subset(copy.deepcopy(dataset), train_idx)
                val_dataset = Subset(copy.deepcopy(dataset), val_idx)
                
                if transforms is not None:
                    train_dataset.dataset.transform = transforms["train"]
                    val_dataset.dataset.transform = transforms["val"]
                
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                
                dataloaders  = {"train":train_dataloader, "val":val_dataloader}
                
                os.makedirs(os.path.join(kfold_root, str(i)),exist_ok=True)

                save_as_pickle(dataloaders,'dataloaders',os.path.join(kfold_root, str(i)))
        
        if transforms is not None:
            test_dataset.transform = transforms["test"]
        print(f"Test: {len(test_dataset)}")
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        randomtest_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        testloaders = {"test": test_dataloader, "randomtest": randomtest_dataloader}
        save_as_pickle(testloaders, "testloaders", kfold_root)
        
        print(f"All: {len(dataset)+len(test_dataset)}")


    return kfold_root


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
