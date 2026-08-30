# -*- encoding: utf-8 -*-
"""OpenCSLR 参数管理器模块。

提供命令行参数解析与配置管理功能，负责解析 CLI 参数、合并 YAML 配置，
并管理训练/测试过程中的所有可配置项。
"""

import argparse
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
            default = './configs/baseline.yaml' ,
            help = 'path to the configuration file' )
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
            default = dict ( ) ,
            help = 'data loader will be used'
        )
        parser.add_argument (
            '--num-worker' ,
            type = int ,
            default = 4 ,
            help = 'the number of worker for data loader' )
        parser.add_argument (
            '--feeder-args' ,
            default = dict ( ) ,
            help = 'the arguments of data loader' )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # model
        parser.add_argument ( '--model' , default = None , help = 'the model will be used' )
        parser.add_argument (
            '--model-args' ,
            type = dict ,
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
        """将配置字典映射到参数解析器，并加载数据集配置。

        将 YAML 配置中的参数更新到解析器的默认值中，然后重新解析参数。
        同时根据 dataset 参数加载对应的数据集配置文件。

        Args:
            config: 配置字典，包含实验配置参数
        """
        key = vars ( getattr(cls, 'ARGS') ).keys ( )
        for k in config.keys ( ) :
            if k not in key :
                print ( 'WRONG ARG: {}'.format ( k ) )
                assert (k in key)
        getattr(cls, 'PARSER').set_defaults ( **config )
        cls.parse()
        dataset_config_path = f"./configs/{ArgumentManager.get ( 'dataset' )}.yaml"
        with open(dataset_config_path, 'r') as f:
            getattr(cls, 'ARGS').dataset_info = yaml.load(f, Loader=yaml.FullLoader)

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
