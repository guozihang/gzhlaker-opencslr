# -*- encoding: utf-8 -*-
"""单阶段训练/评估流水线。

提供训练(eq_train)和评估(eq_eval)的核心循环,以及辅助函数
get_feeder_arg 和 write2file。
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import faulthandler

faulthandler.enable()
import sys
import torch
import torch.distributed as dist
import numpy as np
from manager.evaluation_manager import EvaluationManager
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from manager.log_manager import LogManager
from manager.device_manager import DeviceManager
from manager.dataloader_manager import DataloaderManager
from models.keys import Keys

def get_feeder_arg(cfg, key, default=None):
    """从配置的 feeder_args 中获取指定 key 的值。

    Args:
        cfg: 配置对象,应包含 feeder_args 属性或字典。
        key: 要获取的键名。
        default: 键不存在时返回的默认值,默认为 None。

    Returns:
        feeder_args 中 key 对应的值,若不存在则返回 default。
    """
    feeder_args = getattr(cfg, "feeder_args", {}) or {}
    return feeder_args.get(key, default)

def seq_train(loader, model, optimizer, scheduler, device, epoch_idx, loss_weights=None):
    """执行一个 epoch 的训练。

    使用混合精度训练(AMP),遍历所有 batch 计算损失并反向传播,
    每 200 个 batch 打印一次平均损失,epoch 结束后执行 scheduler.step()。

    Args:
        loader: 训练数据 DataLoader。
        model: CSLR 模型。
        optimizer: 优化器。
        scheduler: 学习率调度器,在每个 epoch 后 step。
        device: 设备(CPU/GPU),当前未直接使用(data 由 DeviceManager 移入设备)。
        epoch_idx: 当前 epoch 索引,用于日志。
        loss_weights: 可选,损失权重,传递给模型内部的损失模块。

    Returns:
        list[float]: 所有 batch 的 loss 值列表。
    """
    model.train()
    loss_value = []
    total_loss_dict = {}    # dict of all types of loss
    clr = [group['lr'] for group in optimizer.param_groups]
    scaler = GradScaler()
    iterator = DataloaderManager.get_iterator("train") if loader is DataloaderManager.get("train") else loader
    for batch_idx, data in enumerate(tqdm(iterator, disable=not DeviceManager.is_main_process())):
        data = {
            Keys.VID: data[0] if isinstance(data[0], torch.Tensor) and data[0].device == DeviceManager.output_device else DeviceManager.to(data[0]),
            Keys.VID_LGT: data[1] if isinstance(data[1], torch.Tensor) and data[1].device == DeviceManager.output_device else DeviceManager.to(data[1]),
            Keys.LABEL: data[2] if isinstance(data[2], torch.Tensor) and data[2].device == DeviceManager.output_device else DeviceManager.to(data[2]),
            Keys.LABEL_LGT: data[3] if isinstance(data[3], torch.Tensor) and data[3].device == DeviceManager.output_device else DeviceManager.to(data[3])
        }
        optimizer.zero_grad()
        with autocast():
            ret_dict = model(data)
            loss = ret_dict[Keys.LOSS]
            loss_dict = ret_dict[Keys.TOTAL_LOSS]
        bad_flag = torch.tensor(
            1.0 if np.isinf(loss.item()) or np.isnan(loss.item()) else 0.0,
            device=DeviceManager.output_device)
        if DeviceManager.is_distributed:
            dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
        if bad_flag.item() > 0.5:
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
    if DeviceManager.is_main_process() and loss_value:
        LogManager.info('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return loss_value

def seq_eval(cfg, loader, model, device, mode, epoch, work_dir):
    """执行一个 epoch 的评估。

    遍历验证集或测试集,收集模型识别结果并写入假设文件,然后调用
    `EvaluationManager.evaluate` 计算 WER(词错误率)。支持跳过帧数超限的 batch 和
    处理 GPU 运行时错误(可选跳过)。

    Args:
        cfg: 配置对象,包含 dataset_info 等评估所需信息。
        loader: 评估数据 DataLoader。
        model: CSLR 模型。
        device: 设备(CPU/GPU),当前未直接使用。
        mode: 评估模式,通常为 "dev" 或 "test",用于日志和输出文件命名。
        epoch: 当前 epoch 索引,用于日志和结果目录。
        work_dir: 工作目录,用于保存输出假设文件和评估结果。

    Returns:
        float: WER(词错误率)百分比,如 25.43 表示 25.43%。
    """
    model.eval()
    if not DeviceManager.is_main_process():
        return 100.0
    total_sent = []
    total_info = []
    max_eval_frames = get_feeder_arg(cfg, "max_eval_frames")
    skip_failed_eval_batches = get_feeder_arg(cfg, "skip_failed_eval_batches", False)
    #save_file = {}
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    iterator = DataloaderManager.get_iterator(mode) if loader is DataloaderManager.get(mode) else loader
    for batch_idx, data in enumerate(tqdm(iterator)):
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
                Keys.VID : data[0] if isinstance(data[0], torch.Tensor) and data[0].device == DeviceManager.output_device else DeviceManager.to(data[0]),
                Keys.VID_LGT : data[1] if isinstance(data[1], torch.Tensor) and data[1].device == DeviceManager.output_device else DeviceManager.to(data[1]),
                Keys.LABEL : data[2] if isinstance(data[2], torch.Tensor) and data[2].device == DeviceManager.output_device else DeviceManager.to(data[2]),
                Keys.LABEL_LGT : data[3] if isinstance(data[3], torch.Tensor) and data[3].device == DeviceManager.output_device else DeviceManager.to(data[3]),
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
        ret = EvaluationManager.evaluate(prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
                                         evaluate_dir=cfg.dataset_info['evaluation_dir'],
                                         evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
                                         output_dir="epoch_{}_result/".format(epoch))
    except Exception as e:
        LogManager.error(f"Unexpected error during evaluation: {e}")
        ret = "Percent Total Error       =  100.00%   (ERROR)"
        return float(ret.split("=")[1].split("%")[0])
    finally:
        pass
    LogManager.info("Epoch {}, {} {}".format(epoch, mode, ret))
    return float(ret.split("=")[1].split("%")[0])

def write2file(path, info, output):
    """将识别结果写入 CTM 格式的文件。

    CTM 格式: `{文件名} 1 {起始时间:.2f} {结束时间:.2f} {词}`,
    每个词均匀分配 0.01 秒的时长。

    Args:
        path: 输出文件路径。
        info: 每个样本的文件名列表,与 output 一一对应。
        output: 识别结果列表,每个元素为词元列表,每个词元形如 (词, 置信度)。
    """
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                f"{info[sample_idx]} 1 {word_idx * 1.0 / 100:.2f} {(word_idx + 1) * 1.0 / 100:.2f} {word[0]}\n"
            )
