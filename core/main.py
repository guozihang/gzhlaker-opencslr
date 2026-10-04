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
    ArgumentManager.init( )
    ConfigManager.init()
    ArgumentManager.map( ConfigManager.get( ) )
    LogManager.init()
    DatasetManager.init()
    CollectManager.init( ArgumentManager.get( ) )
    ModuleManager.init ( )
    DataloaderManager.init()
    DeviceManager.init ( ArgumentManager.get ( "device" ) )
    ExperimentManager.init( ArgumentManager.get( ) )


def run():
    ExperimentManager.run()

def infer(video, video_length):
    return ExperimentManager.run_inference(video, video_length)

if __name__ == "__main__":
    init()
    run()
