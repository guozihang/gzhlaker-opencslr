# -*- encoding: utf-8 -*-
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

class ExperimentManager:
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
        cls.arg = arg
        cls.init_seed()
        cls.init_module()
        cls.load()

    @classmethod
    def init_seed( cls ):
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
        rng_dict = {}
        rng_dict["torch"] = torch.get_rng_state()
        rng_dict["cuda"] = torch.cuda.get_rng_state_all()
        rng_dict["numpy"] = np.random.get_state()
        rng_dict["random"] = random.getstate()
        return rng_dict

    @classmethod
    def set_rng_state(self, rng_dict):
        torch.set_rng_state(rng_dict["torch"])
        torch.cuda.set_rng_state_all(rng_dict["cuda"])
        np.random.set_state(rng_dict["numpy"])
        random.setstate(rng_dict["random"])

    @classmethod
    def init_module( cls ):
        cls.model = ModuleManager.get("model")
        cls.optimizer = ModuleManager.get("optimizer")
        cls.scheduler = ModuleManager.get("scheduler")

    @classmethod
    def run( cls ):
        if cls.arg.phase == 'train':
            cls.run_train()
        elif cls.arg.phase == 'test':
            cls.run_test()

    @classmethod
    def run_train( cls ):
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
        dev_wer = seq_eval ( cls.arg , DataloaderManager.DATALOADER [ "dev" ] , cls.model , cls.device ,
                             "dev" , 6667 , cls.arg.work_dir)
        test_wer = seq_eval ( cls.arg , DataloaderManager.DATALOADER [ "test" ] , cls.model , cls.device ,
                              "test" , 6667 , cls.arg.work_dir)
        LogManager.info ( 'Dev WER: {:05.2f}\n'.format ( dev_wer ) )
        LogManager.info ( 'Test WER: {:05.2f}\n'.format ( test_wer ) )
        LogManager.info ( 'Evaluation Done.\n' )

    @classmethod
    def run_inference(cls, video_data, video_length):
        cls.model.eval()
        with torch.no_grad():
            ret_dict = cls.model(video_data, video_length)
        return ret_dict['recognized_sents']

    @classmethod
    def load(cls):
        if cls.arg.load_weights:
            cls.load_model_weights(cls.model, cls.arg.load_weights)
        elif cls.arg.load_checkpoints:
            cls.load_checkpoint_weights(cls.model, cls.optimizer)
        cls.model = cls.model_to_device(cls.model)

    @classmethod
    def model_to_device(cls, model):
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
        torch.save ( {
            'epoch' : epoch ,
            'model_state_dict' : cls.model.state_dict ( ) ,
            'optimizer_state_dict' : cls.optimizer.state_dict ( ) ,
            'scheduler_state_dict' : cls.scheduler.state_dict ( ) ,
            'rng_state' : cls.save_rng_state ( ) ,
        } , save_path )

    @classmethod
    def load_model_weights(cls, model, weight_path):
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
        state_dict = OrderedDict([(k.replace('.module.', '.'), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    @classmethod
    def load_checkpoint_weights(cls, model, optimizer):
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
