import os

import torch


class DeviceManager(object):
    @classmethod
    def init(cls, device):
        device = str ( device )
        if device != 'None' :
            cls.gpu_list = [ i for i in range ( len ( device.split ( ',' ) ) ) ]
            os.environ [ "CUDA_VISIBLE_DEVICES" ] = device
            output_device = cls.gpu_list [ 0 ]
        cls.output_device = output_device if len ( cls.gpu_list ) > 0 else "cpu"

    @classmethod
    def to(cls, data):
        if isinstance(data, torch.FloatTensor):
            return data.to(cls.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(cls.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(cls.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(cls.output_device)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [cls.to(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))