from typing import Dict, Optional, Any, List
from sklearn.model_selection import KFold, GroupKFold
from torch.utils.data import Dataset, Subset, DataLoader
from ..pickle_io import save_as_pickle, read_as_pickle
import os
import numpy as np
import copy
from abc import abstractmethod


class _BaseKFold():
    def __init__(self, n_splits=5,
                 dataset: Dataset = None,
                 ):
        self.index = []
        self.dataset = dataset
        self.n_splits = n_splits
        self.datasets = []
        self.dataloaders = []

    def get_n_splits(self):
        return self.n_splits
    
    @abstractmethod
    def _split_idx(self, shuffle=False, random_state=None, groups=None):
        pass
    
    def split(self, shuffle=False, random_state=None, groups=None):
        self.index = self._split_idx(shuffle=shuffle, random_state=random_state, groups=groups)
    
    def get_datasets(self, train_transforms=None, val_transforms=None,
                     shuffle=False, random_state=None, groups=None,
                     load_pickle:bool=False, root:str=None):
        kfold_datasets = []
        if not len(self.index) or shuffle:
            if not load_pickle:
                self.split(shuffle=shuffle, random_state=random_state, groups=groups)
            else:
                if root is not None:
                    self.load(root=root)
                else:
                    self.load()
        for train_idx, val_idx in self.index:
            train_dataset = Subset(copy.deepcopy(self.dataset), train_idx)
            val_dataset = Subset(copy.deepcopy(self.dataset), val_idx)
            
            if hasattr(train_dataset.dataset, "transform") and hasattr(val_dataset.dataset, "transform"):
                if train_transforms is not None:
                    train_dataset.dataset.transform = train_transforms
                if val_transforms is not None:
                    val_dataset.dataset.transform = val_transforms
            else:
                raise ValueError(
                    "Dataset object has not attribute 'transform'. "
                )
            trainval_datasets = {"train": train_dataset, "val": val_dataset}
            kfold_datasets.append(trainval_datasets)

        self.datasets = kfold_datasets
        return kfold_datasets
    
    def get_dataloaders(self, batch_size=16, num_workers=2):
        kfold_dataloaders = []
        if not len(self.datasets):
            raise Exception(
                "This instance do not have kfold datasets. "
                "Please execute .get_datasets ."
            )
        else:
            for trainval_datasets in self.datasets:
                train_dataloader = DataLoader(trainval_datasets["train"], shuffle=True,
                                              batch_size=batch_size, num_workers=num_workers)
                val_dataloader = DataLoader(trainval_datasets["val"], shuffle=False,
                                            batch_size=batch_size, num_workers=num_workers)

                trainval_dataloaders = {"train": train_dataloader, "val": val_dataloader}
                kfold_dataloaders.append(trainval_dataloaders)
            self.dataloaders = kfold_dataloaders
            return kfold_dataloaders
    
    def save(self, root = os.path.join(os.getcwd(), "kfold_pickle")):
        """
        Save fold's indices
        """
        print("Save kfold indices...")
        if len(self.index) == self.n_splits:
            os.makedirs(root, exist_ok=True)
            for i in range(self.n_splits):
                os.makedirs(os.path.join(root, str(i)),exist_ok=True)
                save_as_pickle(self.index[i],'index',os.path.join(root, str(i)))
        print("Save completed.")
    
    def load(self, root = os.path.join(os.getcwd(), "kfold_pickle")):
        """
        Load fold's indices
        """
        print("Load kfold indices...")
        for i in range(self.n_splits):
            self.index.append(
                read_as_pickle('index', os.path.join(root, str(i)))
            )
        print("Load completed.")


class KFold_Dataset(_BaseKFold):
    def __init__(self, n_splits=5,
                 dataset: Dataset = None
                 ):
        super().__init__(n_splits=n_splits, dataset=dataset)
        
    def _split_idx(self, shuffle=False, random_state=None, groups=None):
        indices = []
        kf = KFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)
        dataset_idx = list(range(len(self.dataset)))
        print(f"All: {len(dataset_idx)}")
        for (train_idx, val_idx) in kf.split(X=dataset_idx):
            print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
            indices.append([train_idx, val_idx])
        return indices
        

class GroupKFold_Dataset(_BaseKFold):
    def __init__(self, n_splits=5,
                 dataset: Dataset = None):
        super().__init__(n_splits=n_splits, dataset=dataset)
        
    def _split_idx(self, shuffle=False, random_state=None, groups:Any=[]):
        indices = []
        if len(groups) > 0:
            kf = GroupKFold(n_splits=self.n_splits)
            dataset_idx = np.array(range(len(self.dataset)))
            mapping = np.arange(groups.max()+1)
            if shuffle:
                rng = np.random.RandomState(random_state)
                rng.shuffle(mapping)

            print(f"All: {len(dataset_idx)}")
            for train_idx, val_idx in kf.split(X=dataset_idx, groups=mapping[groups]):
                print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
                indices.append([train_idx, val_idx])
            return indices
        else:
            raise ValueError(
                "Attribute groups is empty."
            )
