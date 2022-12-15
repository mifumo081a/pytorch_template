import torch
from ..base import BaseTrainer
from typing import List
import os


def kfold_train(save_model_root: str, trainer_list: List[BaseTrainer], n_splits=5):
    """
    Training per k-fold cross-validation.
    """
    for k in range(n_splits):
        trainer = trainer_list[k]
                
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Fold: ", k)
        trainer.train_loop()

        path = os.path.join(save_model_root, str(k)+"/")
        os.makedirs(path, exist_ok=True)
        trainer.save(path)
        
