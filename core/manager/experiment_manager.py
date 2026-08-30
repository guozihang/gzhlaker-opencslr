# -*- encoding: utf-8 -*-
"""OpenCSLR 实验管理器模块。

负责任务的实验流程编排，包括模型初始化、训练循环、模型评估、
模型保存与加载、断点续训、推理等核心功能。
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import faulthandler
import time
from collections import OrderedDict
faulthandler.enable()
import random
import numpy as np
import torch
import torch.nn as nn
from .log_manager import LogManager
from .dataloader_manager import DataloaderManager
from .module_manager import ModuleManager
from .device_manager import DeviceManager
from pipline.single import seq_train, seq_eval
from models.keys import Keys

class ExperimentManager:
    """实验管理器。

    管理完整的实验生命周期，包括模型初始化、训练、评估、测试、推理，
    以及模型权重的保存与加载、断点续训等功能。
    """
    # 类变量
    arg = None
    rng = None
    device = None
    dataset = {}
    data_loader = {}
    gloss_dict = None
    model = None
    optimizer = None
    kernel_sizes = None
    
    @classmethod
    def init(cls, arg):
        """初始化实验管理器。

        依次执行随机种子初始化、模型模块初始化、模型权重加载。

        Args:
            arg: 命令行参数对象
        """
        cls.arg = arg
        cls.init_seed()
        cls.init_module()
        cls.load()

    @classmethod
    def init_seed( cls ):
        """初始化随机种子。

        如果配置了 random_fix，则固定 Torch、NumPy、Python 内置随机数
        生成器的种子，确保实验结果可复现。
        """
        if cls.arg.random_fix:
            torch.set_num_threads ( 1 )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed ( cls.arg.random_seed )
            torch.cuda.manual_seed_all ( cls.arg.random_seed )
            np.random.seed ( cls.arg.random_seed )
            random.seed ( cls.arg.random_seed )

    @classmethod
    def save_rng_state(self):
        """保存当前随机数生成器状态。

        分别保存 Torch、CUDA、NumPy、Python 内置随机数生成器的状态，
        用于断点续训时恢复随机数序列。

        Returns:
            dict: 包含各随机数生成器状态的字典
        """
        rng_dict = {}
        rng_dict["torch"] = torch.get_rng_state()
        rng_dict["cuda"] = torch.cuda.get_rng_state_all()
        rng_dict["numpy"] = np.random.get_state()
        rng_dict["random"] = random.getstate()
        return rng_dict

    @classmethod
    def set_rng_state(self, rng_dict):
        """恢复随机数生成器状态。

        从保存的状态字典中恢复 Torch、CUDA、NumPy、Python 内置
        随机数生成器的状态，用于断点续训。

        Args:
            rng_dict: 包含各随机数生成器状态的字典
        """
        torch.set_rng_state(rng_dict["torch"])
        torch.cuda.set_rng_state_all(rng_dict["cuda"])
        np.random.set_state(rng_dict["numpy"])
        random.setstate(rng_dict["random"])

    @classmethod
    def init_module( cls ):
        """初始化模型、优化器和学习率调度器。

        从 ModuleManager 中获取已构建的模型、优化器和调度器实例。
        """
        cls.model = ModuleManager.get("model")
        cls.optimizer = ModuleManager.get("optimizer")
        cls.scheduler = ModuleManager.get("scheduler")

    @classmethod
    def run( cls ):
        """运行实验。

        根据 phase 参数决定执行训练或测试流程。
        """
        if cls.arg.phase == 'train':
            cls.run_train()
        elif cls.arg.phase == 'test':
            cls.run_test()

    @classmethod
    def run_train( cls ):
        """执行训练流程。

        按 epoch 循环执行训练和评估，保存最佳模型和定期检查点，
        记录训练时间、验证集和测试集的 WER 指标。
        """
        best_dev = 100.0
        best_epoch = 0
        total_time = 0
        seq_model_list = [ ]
        for epoch in range ( cls.arg.optimizer_args [ 'start_epoch' ] , cls.arg.num_epoch ) :
            save_model = epoch % cls.arg.save_interval == 0
            eval_model = epoch % cls.arg.eval_interval == 0
            epoch_time = time.time ( )
            seq_train (
                DataloaderManager.DATALOADER['train'],
                cls.model ,
                cls.optimizer ,
                cls.scheduler,
                cls.device ,
                epoch ,
                loss_weights = cls.arg.loss_weights
            )
            if eval_model :
                dev_wer = seq_eval (
                    cls.arg ,
                    DataloaderManager.DATALOADER['dev'],
                    cls.model ,
                    cls.device ,
                    'dev' ,
                    epoch ,
                    cls.arg.work_dir
                )
                test_wer = seq_eval (
                    cls.arg ,
                    DataloaderManager.DATALOADER [ 'test' ] ,
                    cls.model ,
                    cls.device ,
                    'test' ,
                    epoch ,
                    cls.arg.work_dir
                )
                LogManager.info ( {"Dev": dev_wer} )
                LogManager.info ( {"Test": test_wer} )
            if dev_wer < best_dev :
                best_dev = dev_wer
                best_epoch = epoch
                model_path = "{}_best_model.pt".format ( cls.arg.work_dir )
                cls.save_model ( epoch , model_path )
                LogManager.info ( 'Save best model' )
            LogManager.info ( 'Best_dev: {:05.2f}, Epoch : {}'.format ( best_dev , best_epoch ) )
            if save_model :
                model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format ( cls.arg.work_dir , dev_wer , epoch )
                seq_model_list.append ( model_path )
                cls.save_model ( epoch , model_path )
            epoch_time = time.time ( ) - epoch_time
            total_time += epoch_time
            LogManager.info ('Epoch {} costs {} mins {} seconds'.format ( epoch , int ( epoch_time ) // 60 ,
                                                                           int ( epoch_time ) % 60 ) )
        LogManager.info ( 'Training costs {} hours {} mins {} seconds'.format ( int ( total_time ) // 60 // 60 ,
                                                                                int ( total_time ) // 60 % 60 ,
                                                                                int ( total_time ) % 60 ) )

    @classmethod
    def run_test( cls ):
        """执行测试流程。

        在验证集和测试集上评估模型，输出 WER 指标。
        """
        dev_wer = seq_eval ( cls.arg , DataloaderManager.DATALOADER [ "dev" ] , cls.model , cls.device ,
                             "dev" , 6667 , cls.arg.work_dir)
        test_wer = seq_eval ( cls.arg , DataloaderManager.DATALOADER [ "test" ] , cls.model , cls.device ,
                              "test" , 6667 , cls.arg.work_dir)
        LogManager.info ( 'Dev WER: {:05.2f}\n'.format ( dev_wer ) )
        LogManager.info ( 'Test WER: {:05.2f}\n'.format ( test_wer ) )
        LogManager.info ( 'Evaluation Done.\n' )

    @classmethod
    def run_inference(cls, video_data, video_length):
        """执行单次推理。

        对输入视频数据进行前向推理，返回识别结果。
        推理时填充 dummy label，loss 结果直接丢弃。

        Args:
            video_data: 视频帧数据张量
            video_length: 视频帧长度张量

        Returns:
            list: 识别结果句子列表
        """
        cls.model.eval()
        data = {
            Keys.VID: DeviceManager.to(video_data),
            Keys.VID_LGT: DeviceManager.to(video_length),
            # forward 固定执行 loss 容器,推理时填充 dummy label,loss 结果直接丢弃
            Keys.LABEL: torch.zeros(1, 1, dtype=torch.long),
            Keys.LABEL_LGT: torch.tensor([1], dtype=torch.long),
        }
        with torch.no_grad():
            ret_dict = cls.model(data)
        return ret_dict[Keys.RECOGNIZED_SENTS]

    @classmethod
    def load(cls):
        """加载模型权重。

        根据配置加载预训练权重或检查点权重，然后将模型迁移到指定设备。
        """
        if cls.arg.load_weights:
            cls.load_model_weights(cls.model, cls.arg.load_weights)
        elif cls.arg.load_checkpoints:
            cls.load_checkpoint_weights(cls.model, cls.optimizer)
        cls.model = cls.model_to_device(cls.model)

    @classmethod
    def model_to_device(cls, model):
        """将模型迁移到指定设备，并为多 GPU 配置 DataParallel。

        将模型移至目标设备，若 GPU 数量大于 1，
        则对 spatial_module_container 应用 DataParallel。

        Args:
            model: 待迁移的模型实例

        Returns:
            nn.Module: 迁移到设备后的模型
        """
        model = model.to(DeviceManager.output_device)
        if len(DeviceManager.gpu_list) > 1:
            model.spatial_module_container = nn.DataParallel(
                model.spatial_module_container,
                device_ids=DeviceManager.gpu_list,
                output_device=DeviceManager.output_device)
        model.cuda()
        return model

    @classmethod
    def save_model ( cls , epoch , save_path ) :
        """保存模型检查点。

        保存当前 epoch、模型状态字典、优化器状态、调度器状态和随机数生成器状态。

        Args:
            epoch: 当前训练轮数
            save_path: 模型保存路径
        """
        torch.save ( {
            'epoch' : epoch ,
            'model_state_dict' : cls.model.state_dict ( ) ,
            'optimizer_state_dict' : cls.optimizer.state_dict ( ) ,
            'scheduler_state_dict' : cls.scheduler.state_dict ( ) ,
            'rng_state' : cls.save_rng_state ( ) ,
        } , save_path )

    @classmethod
    def load_model_weights(cls, model, weight_path):
        """加载预训练模型权重。

        从指定路径加载模型权重，可选择忽略某些权重层。

        Args:
            model: 模型实例
            weight_path: 权重文件路径
        """
        state_dict = torch.load(weight_path)
        if len(cls.arg.ignore_weights):
            for w in cls.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = cls.modified_weights(state_dict['model_state_dict'], False)
        cls.model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        """修改模型权重的键名，移除 DataParallel 添加的 '.module' 前缀。

        Args:
            state_dict: 原始状态字典
            modified: 是否进行额外修改（默认 False）

        Returns:
            OrderedDict: 修改后的状态字典
        """
        state_dict = OrderedDict([(k.replace('.module.', '.'), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    @classmethod
    def load_checkpoint_weights(cls, model, optimizer):
        """加载检查点权重以恢复训练（断点续训）。

        加载模型权重、随机数生成器状态、优化器状态、调度器状态，
        并设置恢复训练的起始 epoch。

        Args:
            model: 模型实例
            optimizer: 优化器实例
        """
        cls.load_model_weights(model, cls.arg.load_checkpoints)
        state_dict = torch.load(cls.arg.load_checkpoints)
        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            LogManager.info("Loading random seeds...")
            cls.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            LogManager.info("Loading optimizer parameters...")
            cls.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        if "scheduler_state_dict" in state_dict.keys():
            LogManager.info("Loading scheduler parameters...")
            cls.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        cls.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        LogManager.info(f"Resuming from checkpoint: epoch {cls.arg.optimizer_args['start_epoch']}")
