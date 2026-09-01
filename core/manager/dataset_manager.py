# -*- encoding: utf-8 -*-
"""Dataset manager module for OpenCSLR.

Handles dataset loading, gloss dictionary management, and dataset instance
creation for training, validation, and test splits.
"""

import importlib

import numpy as np

from .argument_manager import ArgumentManager
from .log_manager import LogManager

class DatasetManager:
    """
    The static class that handle the dataset object.
    """
    DATASET_CLASS = None
    DATASET_OBJECT = {}
    DATASET_LIST = None

    @classmethod
    def init(cls):
        """Initialize dataset instances for all splits.

        Loads the gloss dictionary, creates the dataset class, and instantiates
        dataset objects for train, train_eval, dev, and test modes. Also sets
        the number of classes in model arguments.
        """
        arg = ArgumentManager.get( )
        cls.DATASET_CLASS = import_class(arg.feeder)
        cls.GLOSS_DICT = np.load (
            arg.dataset_info [ 'dict_path' ] ,
            allow_pickle = True
        ).item ()
        cls.DATASET_LIST = list(zip (
            [ "train" , "train_eval" , "dev" , "test" ],
            [ True , False , False , False ]
        ))
        arg.model_args [ 'num_classes' ] = len ( cls.GLOSS_DICT ) + 1

        for idx , (mode , train_flag) in enumerate ( cls.DATASET_LIST ) :
            dataset_arg = arg.feeder_args
            dataset_arg [ "prefix" ] = arg.dataset_info [ 'dataset_root' ]
            for key in ("preprocess_root", "memmap_root", "feature_root", "image_feature_dir",
                        "memmap_layout", "memmap_frame_shape"):
                if key in arg.dataset_info:
                    dataset_arg[key] = arg.dataset_info[key]
            dataset_arg [ "mode" ] = mode.split ( "_" ) [ 0 ]
            dataset_arg [ "transform_mode" ] = train_flag
            dataset_arg [ 'dataset' ] = arg.dataset
            cls.DATASET_OBJECT [ mode ] = cls.DATASET_CLASS (
                gloss_dict = cls.GLOSS_DICT ,
                **dataset_arg
            )
        LogManager.info ( "Loading data finished." )

    @classmethod
    def get(cls, mode="train"):
        """Get the dataset instance for the specified mode.

        Args:
            mode: Dataset mode, one of "train", "train_eval", "dev", "test"

        Returns:
            Dataset: The requested dataset instance
        """
        return cls.DATASET_OBJECT[mode]

    @classmethod
    def get_vocabulary_count(cls):
        """Get the vocabulary size (number of gloss classes plus one for blank).

        Returns:
            int: Vocabulary size
        """
        return len(cls.GLOSS_DICT) + 1

    @classmethod
    def get_dataset_list(cls):
        """Get the list of dataset modes and their training flags.

        Returns:
            list: List of (mode, train_flag) tuples
        """
        return cls.DATASET_LIST

    @classmethod
    def get_gloss_dict(cls):
        """Get the gloss dictionary mapping gloss indices to labels.

        Returns:
            dict: The gloss dictionary
        """
        return cls.GLOSS_DICT


def import_class(name):
    """Dynamically import a class by its fully qualified name.

    Args:
        name: Fully qualified class name, e.g. "dataloader_video.BaseFeeder"

    Returns:
        class: The imported class
    """
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[-2])
    mod = getattr(mod, components[-1])
    return mod
