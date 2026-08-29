# -*- encoding: utf-8 -*-
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import faulthandler

faulthandler.enable()
import sys
import torch
import numpy as np
from libs.slr_eval.wer_calculation import evaluate
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from manager.log_manager import LogManager
from manager.device_manager import DeviceManager
from models.keys import Keys

def get_feeder_arg(cfg, key, default=None):
    feeder_args = getattr(cfg, "feeder_args", {}) or {}
    return feeder_args.get(key, default)

def seq_train(loader, model, optimizer, scheduler, device, epoch_idx, loss_weights=None):
    model.train()
    loss_value = []
    total_loss_dict = {}    # dict of all types of loss
    clr = [group['lr'] for group in optimizer.param_groups]
    scaler = GradScaler()
    for batch_idx, data in enumerate(tqdm(loader)):
        data = {
            Keys.VID: DeviceManager.to(data[0]),
            Keys.VID_LGT: DeviceManager.to(data[1]),
            Keys.LABEL: DeviceManager.to(data[2]),
            Keys.LABEL_LGT: DeviceManager.to(data[3])
        }
        optimizer.zero_grad()
        with autocast():
            ret_dict = model(data)
            loss = ret_dict[Keys.LOSS]
            loss_dict = ret_dict[Keys.TOTAL_LOSS]
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            LogManager.info('loss is nan')
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_value.append(loss.item())
        for item, value in loss_dict.items():
            total_loss_dict[item] = total_loss_dict.get(item, 0) + value
        if batch_idx % 200 == 0:
            LogManager.info(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
            for item, value in total_loss_dict.items():
                LogManager.info(f'\t Mean {item} loss: {value/200:.5f}')
            total_loss_dict = {}
        del ret_dict
        del loss
    scheduler.step()
    LogManager.info('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return loss_value

def seq_eval(cfg, loader, model, device, mode, epoch, work_dir):
    model.eval()
    total_sent = []
    total_info = []
    max_eval_frames = get_feeder_arg(cfg, "max_eval_frames")
    skip_failed_eval_batches = get_feeder_arg(cfg, "skip_failed_eval_batches", False)
    #save_file = {}
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    for batch_idx, data in enumerate(tqdm(loader)):
        batch_info = data[-1]
        batch_shape = tuple(data[0].shape)
        batch_lgt = data[1].tolist()
        if max_eval_frames is not None and len(batch_shape) > 1 and batch_shape[1] > int(max_eval_frames):
            LogManager.info(
                f"Skip {mode} batch {batch_idx}: shape={batch_shape}, "
                f"vid_lgt={batch_lgt}, info={batch_info}"
            )
            continue
        try:
            data = {
                Keys.VID : DeviceManager.to( data [ 0 ] ) ,
                Keys.VID_LGT : DeviceManager.to( data [ 1 ] ) ,
                Keys.LABEL : DeviceManager.to( data [ 2 ] ) ,
                Keys.LABEL_LGT : DeviceManager.to( data [ 3 ] ),
                Keys.INFO: batch_info
            }
            with torch.no_grad():
                ret_dict = model(data)
        except RuntimeError as err:
            LogManager.error(
                f"Eval failed at {mode} batch {batch_idx}: shape={batch_shape}, "
                f"vid_lgt={batch_lgt}, info={batch_info}, error={err}"
            )
            if skip_failed_eval_batches and "illegal memory access" not in str(err).lower():
                torch.cuda.empty_cache()
                continue
            raise

        total_info += [file_name.split("|")[0] for file_name in data[Keys.INFO]]
        total_sent += ret_dict[Keys.RECOGNIZED_SENTS]
    try:
        LogManager.info(work_dir)
        write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        ret = evaluate(prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
                       evaluate_dir=cfg.dataset_info['evaluation_dir'],
                       evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
                       output_dir="epoch_{}_result/".format(epoch))
    except:
        LogManager.error(f"Unexpected error: {sys.exc_info()[0]}")
        ret = "Percent Total Error       =  100.00%   (ERROR)"
        return float(ret.split("=")[1].split("%")[0])
    finally:
        pass
    LogManager.info("Epoch {}, {} {}".format(epoch, mode, ret))
    return float(ret.split("=")[1].split("%")[0])

def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                f"{info[sample_idx]} 1 {word_idx * 1.0 / 100:.2f} {(word_idx + 1) * 1.0 / 100:.2f} {word[0]}\n"
            )
