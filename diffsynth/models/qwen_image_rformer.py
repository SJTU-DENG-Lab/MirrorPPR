from transformers.models.vit.modeling_vit import (
    ViTConfig,
    ViTPreTrainedModel,
    ViTEncoder
)
from torch import nn
import torch
from typing import Optional, Dict, List, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPooling

class RFormerEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
                     
        query_num = config.query_num
        self.query_num = query_num
        self.latent_motion_token = nn.Parameter(torch.zeros(1, query_num, config.hidden_size))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
                                  
        self.projection = nn.Linear(config.input_hidden_size, config.hidden_size, bias=True)
        
                             
        self.position_embeddings = nn.Parameter(torch.randn(1, config.num_patches*2 + 1 + query_num, config.hidden_size))
        self.token_type_embeddings = nn.Parameter(torch.randn(2, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        
        if hasattr(config, "legacy"):
            self.legacy = config.legacy
        else:
            self.legacy = True

    def forward(
        self,
        cond_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, per_seq_length = cond_hidden_states.shape[:2]

        cond_embeddings = self.projection(cond_hidden_states)

        latent_motion_tokens = self.latent_motion_token.expand(batch_size, -1, -1)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        cond_embeddings = torch.cat((latent_motion_tokens, cond_embeddings, sep_tokens), dim=1)

        target_embeddings = self.projection(target_hidden_states)
        embeddings = torch.cat((cond_embeddings, target_embeddings), dim=1)

                                               
        embeddings = embeddings + self.position_embeddings

                                               
        cond_token_type_embeddings = self.token_type_embeddings[0].expand(batch_size, per_seq_length + self.query_num + 1, -1)
        if self.legacy:
            target_token_type_embeddings = self.token_type_embeddings[0].expand(batch_size, per_seq_length, -1)
        else:
            target_token_type_embeddings = self.token_type_embeddings[1].expand(batch_size, per_seq_length, -1)
        token_type_embeddings = torch.cat((cond_token_type_embeddings, target_token_type_embeddings), dim=1)
        embeddings = embeddings + token_type_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class RFormer2DEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        query_num = config.query_num
        self.query_num = query_num
        self.latent_motion_token = nn.Parameter(torch.zeros(1, query_num, config.hidden_size))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.projection = nn.Linear(config.input_hidden_size, config.hidden_size, bias=True)

                                                     
        self.patch_size = 16
        self.max_side_len = 280
        self.d_half = config.hidden_size // 2

                                               
        self.pos_emb_x_cond = nn.Parameter(torch.randn(1, self.max_side_len, self.d_half))
        self.pos_emb_y_cond = nn.Parameter(torch.randn(1, self.max_side_len, self.d_half))
                                   
        self.cls_pos_emb_cond = nn.Parameter(torch.randn(1, 1, config.hidden_size))

                                                 
        self.pos_emb_x_target = nn.Parameter(torch.randn(1, self.max_side_len, self.d_half))
        self.pos_emb_y_target = nn.Parameter(torch.randn(1, self.max_side_len, self.d_half))
                                     
        self.cls_pos_emb_target = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
                                     
        self.query_pos_embedding = nn.Parameter(torch.randn(1, query_num, config.hidden_size))
        self.sep_pos_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_size))
                                                                     

        self.token_type_embeddings = nn.Parameter(torch.randn(2, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        if hasattr(config, "legacy"):
            self.legacy = config.legacy
        else:
            self.legacy = True

    def _get_grid_embedding(self, pos_emb_x, pos_emb_y, h, w):
        """辅助函数：生成纯粹的 2D 网格位置编码 (不含 CLS)"""
               
        y_emb = pos_emb_y[:, :h, :]              
        x_emb = pos_emb_x[:, :w, :]              

                   
        y_grid = y_emb.unsqueeze(2).expand(-1, -1, w, -1)                 
        x_grid = x_emb.unsqueeze(1).expand(-1, h, -1, -1)                 
        
                    
        grid_emb_2d = torch.cat([y_grid, x_grid], dim=-1)               
        return grid_emb_2d[0].flatten(0, 1)           

    def forward(
        self,
        cond_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        sample1_shapes: List[torch.Tensor] = None 
    ) -> torch.Tensor:
        batch_size = cond_hidden_states.shape[0]

                   
        cond_embeddings = self.projection(cond_hidden_states)
        target_embeddings = self.projection(target_hidden_states)

                              
                                                                   
        latent_motion_tokens = self.latent_motion_token.expand(batch_size, -1, -1)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        
                                                   
        cond_part = torch.cat((latent_motion_tokens, cond_embeddings, sep_tokens), dim=1)
        embeddings = torch.cat((cond_part, target_embeddings), dim=1)

                                       
        
        batch_pixel_h = sample1_shapes[0]
        batch_pixel_w = sample1_shapes[1]

        batch_pos_embeddings = []

        for b in range(batch_size):
                              
            pixel_h = int(batch_pixel_h[b].item())
            pixel_w = int(batch_pixel_w[b].item())
            h = pixel_h // self.patch_size
            w = pixel_w // self.patch_size
            num_patches = h * w

                                   
                                                      
                                       
            len_cond = cond_hidden_states.shape[1]
            has_cls_cond = (len_cond == num_patches + 1)
            
                    
            grid_cond = self._get_grid_embedding(self.pos_emb_x_cond, self.pos_emb_y_cond, h, w)
            
                            
            if has_cls_cond:
                                        
                                          
                pos_emb_cond = torch.cat([self.cls_pos_emb_cond[0], grid_cond], dim=0)
            else:
                pos_emb_cond = grid_cond


                                     
            len_target = target_hidden_states.shape[1]
            has_cls_target = (len_target == num_patches + 1)
            
            grid_target = self._get_grid_embedding(self.pos_emb_x_target, self.pos_emb_y_target, h, w)
            
            if has_cls_target:
                                        
                                            
                pos_emb_target = torch.cat([self.cls_pos_emb_target[0], grid_target], dim=0)
            else:
                pos_emb_target = grid_target


                            
                                                                                   
            full_pos_emb = torch.cat([
                self.query_pos_embedding[0],         
                pos_emb_cond,                                    
                self.sep_pos_embedding[0],         
                pos_emb_target                                     
            ], dim=0)

            batch_pos_embeddings.append(full_pos_emb)

               
        pos_embeddings = torch.stack(batch_pos_embeddings).to(embeddings.device)
        embeddings = embeddings + pos_embeddings
                                                     

                                               
        len_cond_total = latent_motion_tokens.shape[1] + cond_embeddings.shape[1] + sep_tokens.shape[1]
        len_target_total = target_embeddings.shape[1]
        
                                              
        cond_token_type_embeddings = self.token_type_embeddings[0].expand(batch_size, len_cond_total, -1)
        
                                              
        if self.legacy:
            target_token_type_embeddings = self.token_type_embeddings[0].expand(batch_size, len_target_total, -1)
        else:
            target_token_type_embeddings = self.token_type_embeddings[1].expand(batch_size, len_target_total, -1)
            
        token_type_embeddings = torch.cat((cond_token_type_embeddings, target_token_type_embeddings), dim=1)
        embeddings = embeddings + token_type_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class ViTPooler(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
                                                                             
                             
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RFormer(ViTPreTrainedModel):
    def __init__(self, add_pooling_layer: bool = False):                   
        
                      
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            qkv_bias=True,
            model_type="vit",
                                                              
            query_num=8,
            input_hidden_size=1024,
            num_patches=4097,                          
            legacy=True         
        )

                                  
        super().__init__(config)
        self.config = config
        
                  
        self.query_num = config.query_num
        self.embeddings = RFormerEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

                                                       
        self.post_init()

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, RFormerEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.token_type_embeddings.data = nn.init.trunc_normal_(
                module.token_type_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.token_type_embeddings.dtype)

            module.latent_motion_token.data = nn.init.trunc_normal_(
                module.latent_motion_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.latent_motion_token.dtype)

            module.sep_token.data = nn.init.trunc_normal_(
                module.sep_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.sep_token.dtype)

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        cond_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states
        )
        
                               
                           
               
                                    
                              
                                                                       
                

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class RFormer2D(ViTPreTrainedModel):
    def __init__(self, add_pooling_layer: bool = False):                   
        
        config = ViTConfig(
            attn_implementation="sdpa",
            dtype="bfloat16",
            output_attentions=False,                          
            output_hidden_states=False,             
            return_dict=True,
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            qkv_bias=True,
            model_type="vit",
                                                              
            query_num=8,
            input_hidden_size=1024,
            num_patches=16801,                          
            legacy=True         
        )

        super().__init__(config)
        self.config = config
        self.query_num = config.query_num
        self.embeddings = RFormer2DEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        self.post_init()

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, RFormer2DEmbeddings):
                                               
            
                          
            def init_param(param):
                param.data = nn.init.trunc_normal_(
                    param.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(param.dtype)

                         
            init_param(module.pos_emb_x_cond)
            init_param(module.pos_emb_y_cond)

                                   
            init_param(module.pos_emb_x_target)
            init_param(module.pos_emb_y_target)

                             
            init_param(module.query_pos_embedding)
            init_param(module.sep_pos_embedding)
                                                         

            init_param(module.token_type_embeddings)
            init_param(module.latent_motion_token)
            init_param(module.sep_token)
            
            init_param( module.cls_pos_emb_cond)
            init_param( module.cls_pos_emb_target)


    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        cond_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        sample1_shapes: List[torch.Tensor], 
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
                                            
        
        
                                                                                                                                                

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states,
            sample1_shapes=sample1_shapes 
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask
        )
        sequence_output = encoder_outputs.last_hidden_state             
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

                             
                                                                                                                  
                                                       

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

        
                   
                                                          
                                     
                                                             

                   
                  
               
                                                  
                                 
                         

                        
              
                                         
                                 
                                                                                                   
