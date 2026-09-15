# -*- encoding: utf-8 -*-
"""Module manager module for OpenCSLR.

Handles the construction and lifecycle of model, optimizer, and learning rate
scheduler objects for experiments.
"""

import torch.optim as optim

from .argument_manager import ArgumentManager
from .log_manager import LogManager
from .dataset_manager import DatasetManager
from .device_manager import DeviceManager

class ModuleManager:
    """
    The static class that handle all the module object and epoch datawf
    """
    MODEL_OBJECT = None
    OPTIMIZER_OBJECT = None
    SCHEDULER_OBJECT = None

    @classmethod
    def init(cls):
        """Initialize all modules: model, optimizer, and scheduler.

        Calls the initialization methods for model, optimizer, and scheduler
        in order, then prints the model architecture for logging.
        """
        cls.init_model_object ( )
        cls.init_optimizer_object()
        cls.init_scheduler_object()
        cls.print_model_object( )

    @classmethod
    def init_model_object(cls):
        """Build the model object using the configured build function.

        Loads the model build function from the model configuration, constructs
        the model with the given arguments, gloss dictionary, and loss weights,
        then moves the model to CUDA.
        """
        arg = ArgumentManager.get ( )
        build_function = cls.load( arg.model )
        arg.model_args[ 'decode_mode' ] = arg.decode_mode
        ModuleManager.MODEL_OBJECT = build_function (
            arg.model_args ,
            gloss_dict = DatasetManager.get_gloss_dict ( ) ,
            loss_weights = arg.loss_weights ,
        )
        ModuleManager.MODEL_OBJECT = ModuleManager.MODEL_OBJECT.to(DeviceManager.output_device)

    @classmethod
    def init_optimizer_object(cls):
        """Initialize the optimizer object.

        Creates an SGD or Adam optimizer based on the configuration.

        Raises:
            ValueError: When the optimizer type is not supported
        """
        arg = ArgumentManager.get( ).optimizer_args
        if arg [ "optimizer" ] == 'SGD' :
            cls.OPTIMIZER_OBJECT = optim.SGD (
                cls.MODEL_OBJECT.parameters() ,
                lr = arg [ 'base_lr' ] ,
                momentum = 0.9 ,
                nesterov = arg [ 'nesterov' ] ,
                weight_decay = arg [ 'weight_decay' ]
            )
        elif arg [ "optimizer" ] == 'Adam' :
            cls.OPTIMIZER_OBJECT = optim.Adam (
                cls.MODEL_OBJECT.parameters ( ) ,
                lr = arg [ 'base_lr' ] ,
                weight_decay = arg [ 'weight_decay' ]
            )
        else :
            raise ValueError ( f"Unknown optimizer: {arg['optimizer']!r}" )


    @classmethod
    def init_scheduler_object(cls):
        """Initialize the learning rate scheduler.

        Creates a MultiStepLR scheduler with the configured step milestones
        and gamma value. The optimizer type was already validated by
        init_optimizer_object, so no branch is needed here.
        """
        arg = ArgumentManager.get ( ).optimizer_args
        cls.SCHEDULER_OBJECT = optim.lr_scheduler.MultiStepLR(
            cls.OPTIMIZER_OBJECT, milestones=arg['step'], gamma=0.2)

    @classmethod
    def print_model_object(cls):
        """Print the model architecture as a rich panel for logging.

        Extracts the model name from the configuration and logs the model
        structure using LogManager.
        """
        if DeviceManager.is_main_process():
            model_name = ArgumentManager.get( "model" )
            LogManager.info_panel( cls.MODEL_OBJECT , title= f"{model_name}" )

    @classmethod
    def get(cls, module_name=None):
        """Get the requested module object.

        Args:
            module_name: Module identifier, one of "model", "optimizer", "scheduler"

        Returns:
            The requested module object (model, optimizer, or scheduler)

        Raises:
            ValueError: When module_name is None or unknown
        """
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
        """Set the model to training or evaluation mode.

        Args:
            mode: "train" for training mode or "eval" for evaluation mode

        Raises:
            ValueError: When the mode is not "train" or "eval"
        """
        if mode == "train":
            cls.MODEL_OBJECT.train()
        elif mode == "eval":
            cls.MODEL_OBJECT.eval()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def load(cls, name):
        """按注册名获取模型构建函数。

        注册表由 models/ 下各构建函数的 @register_model 装饰器填充,新增模型
        只需在 config 的 ``model:`` 字段写上注册名。

        Args:
            name: 注册名(如 "slowfast")

        Returns:
            callable: 模型构建函数。

        Raises:
            ValueError: 注册名不存在时抛出,并列出全部可用模型。
        """
        from models import get_model_builder  # 导入 models 即触发各模型注册
        return get_model_builder(name)