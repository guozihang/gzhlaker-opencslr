import torch
import torch.nn as nn
from typing import Dict , Any , Tuple , Optional , Union , List


class Container ( nn.Module ) :
    def __init__ ( self , modules: Union [ List , Dict ] ) :
        super ( Container , self ).__init__ ( )
        self.module_list = nn.ModuleList ( modules )

    def forward ( self , data: Dict [ str , torch.Tensor ] ) -> Dict [ str , torch.Tensor ] :
        # 如果是ModuleList
        if hasattr ( self , 'module_list' ) :
            for module in self.module_list :
                output = module ( data )
                if isinstance ( output , dict ) :
                    data.update ( output )
        # 如果是字典形式的模块
        elif hasattr ( self , 'module_names' ) :
            for name in self.module_names :
                module = getattr ( self , name )
                output = module ( data )
                if isinstance ( output , dict ) :
                    data.update ( output )
        return data


class SignLanguageModel ( nn.Module ) :
    def __init__ ( self , spatial_module_container: Container , temporal_module_container: Container ,
                   loss_module_container: Container, decoder) :
        super ( ).__init__ ( )
        self.spatial_module_container = spatial_module_container
        self.temporal_module_container = temporal_module_container
        self.loss_module_container = loss_module_container
        self.decoder = decoder
        self.register_backward_hook ( self.backward_hook )

    def backward_hook ( self , module , grad_input , grad_output ) :
        for g in grad_input :
            g [ g != g ] = 0

    def forward ( self , data: Dict [ str , torch.Tensor ] ) -> Dict [ str , Any ] :
        data.update ( self.spatial_module_container ( data ) )
        data.update ( self.temporal_module_container ( data ) )
        data.update ( self.loss_module_container ( data ) )
        data.update ( self.decoder ( data ) )
        return data