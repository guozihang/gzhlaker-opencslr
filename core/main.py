"""OpenCSLR 训练与推理入口。

按照初始化链依次初始化各个管理器模块（Argument -> Config -> Device -> Log
-> Dataset -> Collect -> Module -> Dataloader -> Experiment），然后启动训练
或测试流程。支持通过 torchrun 启动分布式训练。
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
torch.backends.cudnn.enabled = False

from manager.argument_manager import ArgumentManager
from manager.config_manager import ConfigManager
from manager.log_manager import LogManager
from manager.experiment_manager import ExperimentManager
from manager.dataset_manager import DatasetManager
from manager.dataloader_manager import DataloaderManager
from manager.module_manager import ModuleManager
from manager.device_manager import DeviceManager
from manager.collect_manager import CollectManager


def init():
    """初始化所有管理器模块。

    按固定顺序依次初始化：ArgumentManager -> ConfigManager -> DeviceManager
    -> LogManager -> DatasetManager -> CollectManager -> ModuleManager
    -> DataloaderManager -> ExperimentManager。
    """
    ArgumentManager.init( )
    ConfigManager.init()
    ArgumentManager.map( ConfigManager.get( ) )
    DeviceManager.init ( ArgumentManager.get ( "device" ) )
    LogManager.init()
    DatasetManager.init()
    CollectManager.init( ArgumentManager.get( ) )
    ModuleManager.init ( )
    DataloaderManager.init()
    ExperimentManager.init( ArgumentManager.get( ) )


def run():
    """启动训练或测试流程。

    通过 ExperimentManager 运行指定配置的训练、验证和测试。
    """
    try:
        ExperimentManager.run()
        DeviceManager.barrier()
    finally:
        DataloaderManager.shutdown()
        DeviceManager.cleanup()

def infer(video, video_length):
    """对输入视频进行推理。

    Args:
        video: 输入视频数据。
        video_length: 视频长度（帧数）。

    Returns:
        模型推理结果。
    """
    return ExperimentManager.run_inference(video, video_length)

if __name__ == "__main__":
    init()
    run()
