# -*- encoding: utf-8 -*-
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
        return cls.DATASET_OBJECT[mode]

    @classmethod
    def get_vocabulary_count(cls):
        return len(cls.GLOSS_DICT) + 1

    @classmethod
    def get_dataset_list(cls):
        return cls.DATASET_LIST

    @classmethod
    def get_gloss_dict(cls):
        return cls.GLOSS_DICT


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[-2])
    mod = getattr(mod, components[-1])
    return mod
