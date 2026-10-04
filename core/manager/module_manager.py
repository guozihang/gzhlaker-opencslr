# -*- encoding: utf-8 -*-
import importlib

import torch.optim as optim

from .argument_manager import ArgumentManager
from .log_manager import LogManager
from .dataset_manager import DatasetManager
from .collect_manager import CollectManager
from libs.sync_batchnorm import convert_model

class ModuleManager:
    """
    The static class that handle all the module object and epoch datawf
    """
    MODEL_OBJECT = None
    OPTIMIZER_OBJECT = None
    SCHEDULER_OBJECT = None

    @classmethod
    def init(cls):
        cls.init_model_object ( )
        cls.init_optimizer_object()
        cls.init_scheduler_object()
        cls.print_model_object( )

    @classmethod
    def init_model_object(cls):
        arg = ArgumentManager.get ( )
        build_function = cls.load( arg.model )
        ModuleManager.MODEL_OBJECT= build_function (
            arg.model_args ,
            gloss_dict = DatasetManager.get_gloss_dict ( ) ,
            loss_weights = arg.loss_weights ,
        )
        # ModuleManager.MODEL_OBJECT = convert_model ( ModuleManager.MODEL_OBJECT )
        ModuleManager.MODEL_OBJECT.to("cuda")

    @classmethod
    def init_optimizer_object(cls, optimizer_type="Adam"):
        arg = ArgumentManager.get( ).optimizer_args
        if arg [ "optimizer" ] == 'SGD' :
            cls.OPTIMIZER_OBJECT = optim.SGD (
                cls.MODEL_OBJECT ,
                lr = arg [ 'base_lr' ] ,
                momentum = 0.9 ,
                nesterov = arg [ 'nesterov' ] ,
                weight_decay = arg [ 'weight_decay' ]
            )
        elif arg [ "optimizer" ] == 'Adam' :
            alpha = arg [ 'learning_ratio' ]
            cls.OPTIMIZER_OBJECT = optim.Adam (
                cls.MODEL_OBJECT.parameters ( ) ,
                lr = arg [ 'base_lr' ] ,
                weight_decay = arg [ 'weight_decay' ]
            )
        else :
            raise ValueError ( )


    @classmethod
    def init_scheduler_object(cls):
        arg = ArgumentManager.get ( ).optimizer_args
        if arg["optimizer"] in ['SGD', 'Adam']:
            cls.SCHEDULER_OBJECT = optim.lr_scheduler.MultiStepLR(cls.OPTIMIZER_OBJECT, milestones=arg['step'], gamma=0.2)
        else:
            raise ValueError()

    @classmethod
    def print_model_object(cls):
        model_name = ArgumentManager.get( "model" ).split ( "." )[-2 ]
        LogManager.info_panel( cls.MODEL_OBJECT , title= f"{model_name}" )

    @classmethod
    def get(cls, module_name=None):
        if module_name is None:
            raise ValueError("Module name must be provided")
        elif module_name == "model":
            return ModuleManager.MODEL_OBJECT
        elif module_name == "optimizer":
            return ModuleManager.OPTIMIZER_OBJECT
        elif module_name == "scheduler":
            return ModuleManager.SCHEDULER_OBJECT
        # elif module_name == "scaler":
        #     return ModuleManager.SCALER_OBJECT
        else:
            raise ValueError(f"Unknown module name: {module_name}")

    @classmethod
    def set( cls, mode ):
        if mode == "train":
            cls.MODEL_OBJECT.train()
        elif mode == "eval":
            cls.MODEL_OBJECT.eval()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def load(cls, name):
        components = name.rsplit ( '.' , 1 )
        mod = importlib.import_module ( components [ -2 ] )
        mod = getattr ( mod , components [ -1 ] )
        return mod