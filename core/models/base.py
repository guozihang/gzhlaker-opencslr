import torch
import torch.nn as nn
from typing import Dict , Any , List


class Container ( nn.Module ) :
    def __init__ ( self , modules : List [ nn.Module ] ) :
        super ( Container , self ).__init__ ( )
        self.module_list = nn.ModuleList ( modules )

    def forward ( self , data : Dict [ str , Any ] ) -> Dict [ str , Any ] :
        for module in self.module_list :
            output = module ( data )
            if not isinstance ( output , dict ) :
                raise TypeError (
                    "Container 中的 {} 必须返回 dict,实际返回 {}".format (
                        module.__class__.__name__ , type ( output ).__name__ ) )
            data.update ( output )
        return data


class SignLanguageModel ( nn.Module ) :
    def __init__ ( self , spatial_module_container : Container , temporal_module_container : Container ,
                   loss_module_container : Container, decoder ) :
        super ( ).__init__ ( )
        self.spatial_module_container = spatial_module_container
        self.temporal_module_container = temporal_module_container
        self.loss_module_container = loss_module_container
        self.decoder = decoder

    def forward ( self , data : Dict [ str , Any ] ) -> Dict [ str , Any ] :
        # 各容器原地更新 data 并返回同一字典,无需重复 update
        self.spatial_module_container ( data )
        self.temporal_module_container ( data )
        self.loss_module_container ( data )
        output = self.decoder ( data )
        if not isinstance ( output , dict ) :
            raise TypeError (
                "decoder {} 必须返回 dict,实际返回 {}".format (
                    self.decoder.__class__.__name__ , type ( output ).__name__ ) )
        data.update ( output )
        return data
