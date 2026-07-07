import torch
from torch import nn

from typing import Optional

class QwenImageMAE(nn.Module):
    def __init__(
        self, 
    ):
        super().__init__()
        from transformers import ViTMAEConfig, ViTMAEModel
        
        config=ViTMAEConfig(**{
            "architectures": [
                "ViTMAEForPreTraining"
            ],
            "attention_probs_dropout_prob": 0.0,
            "decoder_hidden_size": 512,
            "decoder_intermediate_size": 2048,
            "decoder_num_attention_heads": 16,
            "decoder_num_hidden_layers": 8,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 1024,
            "image_size": 224,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-12,
            "mask_ratio": 0.0,    
            "model_type": "vit_mae",
            "norm_pix_loss": False,
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 24,
            "patch_size": 16,
            "qkv_bias": True,
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa"
            }
            )
        
        self.model = ViTMAEModel(config)
        self.config=config


    def forward(self, pixel_values):
        outputs=self.model(pixel_values,interpolate_pos_encoding=True)
        return outputs.last_hidden_state
    
    def new_forward(self, pixel_values,sample1_shapes):
        outputs=self.model.new_forward(pixel_values,sample1_shapes,interpolate_pos_encoding=True)
        return outputs.last_hidden_state

                                              
                                            
                                    
                        
                               
                                                                     

                        
                                                                



                                                  
                                                            
                                    
                                               

                                    
                                     
                               


                         
                                                                      
                                   
   


    
                                         

                                                                                                          
                                                                                           

                  
                                                  

                   
        

              
                                         

                
                          
                                                                      
                        
   

                      
                                                                                             

               
