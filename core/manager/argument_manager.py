# -*- encoding: utf-8 -*-
"""OpenCSLR 参数管理器模块。

提供命令行参数解析与配置管理功能，负责解析 CLI 参数、合并 YAML 配置，
并管理训练/测试过程中的所有可配置项。
"""

import argparse
import copy
import json
import os
import yaml


class ArgumentManager:
    """参数管理器。

    使用类方法管理所有参数，维护全局参数解析器(PARSER)和解析结果(ARGS)。
    支持通过 CLI 参数和 YAML 配置两种方式设置实验参数。
    """

    @classmethod
    def init(cls):
        """初始化参数解析器，注册所有 CLI 参数并解析。

        创建 ArgumentParser 实例，注册所有支持的命令行参数（包括工作目录、
        配置文件、随机种子、设备、训练参数、数据集参数、模型参数、优化器参数等），
        然后调用 parse() 执行解析。
        """
        parser = argparse.ArgumentParser (
            description = 'The pytorch implementation for Visual Alignment Constraint '
                          'for Continuous Sign Language Recognition.' )
        parser.add_argument (
            '--work-dir' ,
            default = './work_dir/temp' ,
            help = 'the work folder for storing results' )
        parser.add_argument (
            '--config' ,
            default = './configs/exp.yaml' ,
            help = 'path to the experiment configuration file' )
        parser.add_argument (
            '--exp' ,
            default = 'baseline' ,
            help = 'experiment name, a section in the exp config file' )
        parser.add_argument (
            '--random_fix' ,
            type = str2bool ,
            default = True ,
            help = 'fix random seed or not' )
        parser.add_argument (
            '--device' ,
            type = str ,
            default = 0 ,
            help = 'the indexes of GPUs for training or testing' )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # processor
        parser.add_argument (
            '--phase' , default = 'train' , help = 'can be train, test and features' )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # debug
        parser.add_argument (
            '--save-interval' ,
            type = int ,
            default = 200 ,
            help = 'the interval for storing models (#epochs)' )
        parser.add_argument (
            '--random-seed' ,
            type = int ,
            default = 0 ,
            help = 'the default value for random seed.' )
        parser.add_argument (
            '--eval-interval' ,
            type = int ,
            default = 100 ,
            help = 'the interval for evaluating models (#epochs)' )
        parser.add_argument (
            '--print-log' ,
            type = str2bool ,
            default = True ,
            help = 'print logging or not' )
        parser.add_argument (
            '--log-interval' ,
            type = int ,
            default = 20 ,
            help = 'the interval for printing messages (#iteration)' )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # feeder
        parser.add_argument (
            '--feeder' , default = 'dataloader_video.BaseFeeder' , help = 'data loader will be used' )
        parser.add_argument (
            '--dataset' ,
            default = None ,
            help = 'data loader will be used'
        )
        parser.add_argument (
            '--dataset-info' ,
            type = json_dict,
            default = dict ( ) ,
            help = 'data loader will be used'
        )
        parser.add_argument (
            '--num-worker' ,
            type = int ,
            default = 4 ,
            help = 'the number of worker for data loader' )
        parser.add_argument('--eval-num-worker', type=int, default=0,
                            help='number of workers for evaluation loaders')
        parser.add_argument('--prefetch-factor', type=int, default=2,
                            help='batches prefetched by each worker')
        parser.add_argument('--persistent-workers', type=str2bool, default=True,
                            help='keep training DataLoader workers alive between epochs')
        parser.add_argument('--pin-memory', type=str2bool, default=True,
                            help='pin host batches for asynchronous CUDA copies')
        parser.add_argument('--worker-threads', type=int, default=1,
                            help='number of torch/OpenCV threads per DataLoader worker')
        parser.add_argument('--preopen-memmap', type=str2bool, default=True,
                            help='open the training memmap before workers are forked')
        parser.add_argument('--gpu-prefetch', type=str2bool, default=True,
                            help='prefetch and transform batches on a CUDA stream')
        parser.add_argument('--length-bucket-size', type=int, default=0,
                            help='length bucket width in batches; 0 disables bucketing')
        parser.add_argument (
            '--feeder-args' ,
            type = json_dict,
            default = dict ( ) ,
            help = 'the arguments of data loader' )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # model
        parser.add_argument ( '--model' , default = None , help = 'the model will be used' )
        parser.add_argument (
            '--model-args' ,
            type = json_dict,
            default = dict ( ) ,
            help = 'the arguments of model' )
        parser.add_argument (
            '--load-weights' ,
            default = None ,
            help = 'load weights for network initialization' )
        parser.add_argument (
            '--load-checkpoints' ,
            default = None ,
            help = 'load checkpoints for continue training' )
        parser.add_argument (
            '--decode-mode' ,
            default = "max" ,
            help = 'search mode for decode, max or beam' )
        parser.add_argument (
            '--ignore-weights' ,
            type = str ,
            default = [ ] ,
            nargs = '+' ,
            help = 'the name of weights which will be ignored in the initialization' )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # optim
        parser.add_argument (
            '--batch-size' , type = int , default = 16 , help = 'training batch size' )
        parser.add_argument (
            '--test-batch-size' , type = int , default = 8 , help = 'test batch size' )

        default_optimizer_dict = {
            "base_lr" : 1e-2 ,
            "optimizer" : "SGD" ,
            "nesterov" : False ,
            "step" : [ 5 , 10 ] ,
            "weight_decay" : 0.00005 ,
            "start_epoch" : 1 ,
        }
        default_loss_dict = {
            "SeqCTC" : 1.0 ,
        }

        default_wandb_dict = {
            "enable" : True,
            "project" : "openslr",
            "entity" : "gzhlaker",

        }

        parser.add_argument (
            '--loss-weights' ,
            default = default_loss_dict ,
            help = 'loss selection'
        )

        parser.add_argument (
            '--optimizer-args' ,
            default = default_optimizer_dict ,
            help = 'the arguments of optimizer' )

        parser.add_argument (
            '--wandb' ,
            default = default_wandb_dict ,
            help = 'the arguments of wandb' )

        parser.add_argument (
            '--num-epoch' ,
            type = int ,
            default = 80 ,
            help = 'stop training in which epoch' )

        if not hasattr( cls , 'PARSER' ) :
            setattr ( cls , 'PARSER' , parser )

        cls.parse()

    @classmethod
    def parse( cls ):
        """解析命令行参数。

        调用 ArgumentParser 解析命令行参数，并将结果存储为类属性 ARGS。

        Returns:
            argparse.Namespace: 解析后的参数对象
        """
        setattr ( cls , 'ARGS' , cls.get_parser ( ).parse_args ( ) )
        return getattr ( cls , 'ARGS' )

    @classmethod
    def get(cls, argument=None):
        """获取参数值。

        Args:
            argument: 参数名。若为 None，则返回全部参数对象；否则返回指定参数的值。

        Returns:
            全部参数对象或指定参数的值
        """
        if argument is None:
            return getattr(cls, 'ARGS')
        else:
            return getattr(getattr(cls, 'ARGS'), argument)

    @classmethod
    def get_parser(cls):
        """获取参数解析器。

        Returns:
            argparse.ArgumentParser: 全局参数解析器实例
        """
        return getattr(cls, 'PARSER')

    @classmethod
    def map( cls, config):
        """将 YAML 配置深度合并到默认值，并保留 CLI 覆盖优先级。

        配置优先级为：命令行参数 > 实验 YAML > 代码默认值。
        """
        if not isinstance(config, dict):
            raise TypeError("experiment config must be a mapping")

        argument_names = set(vars(cls.get()))
        unknown = set(config) - argument_names
        if unknown:
            raise ValueError(f"Unknown argument(s): {', '.join(sorted(unknown))}")

        cli_values = vars(cls.get_parser().parse_args())
        defaults = vars(cls.get_parser().parse_args([]))
        merged = cls._deep_merge(defaults, config)
        merged = cls._restore_cli_values(merged, cli_values, defaults)
        cls.get_parser().set_defaults(**merged)
        cls.parse()

        dataset = cls.get("dataset")
        if not dataset:
            raise ValueError("dataset must be configured before loading dataset metadata")
        config_dir = os.path.dirname(os.path.abspath(cls.get("config")))
        dataset_config_path = os.path.join(config_dir, "dataset.yaml")
        with open(dataset_config_path, "r", encoding="utf-8") as reader:
            all_dataset_info = yaml.safe_load(reader) or {}
        if not isinstance(all_dataset_info, dict):
            raise TypeError("dataset config must be a mapping of dataset name -> config")
        dataset_info = all_dataset_info.get(dataset)
        if dataset_info is None:
            raise ValueError(
                f"Unknown dataset {dataset!r}: not found in {dataset_config_path}. "
                f"Available: {sorted(all_dataset_info)}."
            )
        if not isinstance(dataset_info, dict):
            raise TypeError("dataset config must be a mapping")
        cls.ARGS.dataset_info = cls._deep_merge(cls.ARGS.dataset_info, dataset_info)

    @staticmethod
    def _deep_merge(base, override):
        """递归合并字典；嵌套字典逐层合并，其它值由 override 替换。"""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(result.get(key), dict) and isinstance(value, dict):
                result[key] = ArgumentManager._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def _restore_cli_values(merged, cli_values, parser_defaults):
        """只恢复实际出现在命令行中的参数，避免默认值覆盖 YAML。"""
        import sys
        cli_names = {item.lstrip("-").replace("-", "_").split("=")[0]
                     for item in sys.argv[1:] if item.startswith("-")}
        for name, value in cli_values.items():
            if name in cli_names and value != parser_defaults.get(name):
                merged[name] = value
        return merged

def json_dict(v):
    """将 JSON 字符串解析为字典。

    Args:
        v: JSON 格式字符串

    Returns:
        dict: 解析后的字典
    """
    return json.loads(v)

def str2bool(v):
    """将字符串转换为布尔值。

    Args:
        v: 输入字符串

    Returns:
        bool: 转换后的布尔值

    Raises:
        argparse.ArgumentTypeError: 当输入无法转换为布尔值时抛出
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
