import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed

                                       

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    mrope_section: list[int],                                                           
    unsqueeze_dim: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    [重写] 严格按照 Qwen2.5-VL 的实现。
    
    将 3D RoPE (T, H, W) 应用于 query 和 key。

    Args:
        q (`torch.Tensor`): query (B, N_img, H, Hc)
        k (`torch.Tensor`): key (B, N_img, H, Hc)
        cos (`torch.Tensor`): cosine (3, B, N_img, Hc)
        sin (`torch.Tensor`): sine (3, B, N_img, Hc)
        mrope_section (`List[int]`): 
            T, H, W 的通道维度列表, e.g., [16, 24, 24].
            注意: 2 * sum(mrope_section) 必须等于 Hc.
        unsqueeze_dim (`int`, *optional*, defaults to 2):
            为 (cos, sin) 增加的广播维度。
            在我们的 Attention 模块中, q/k 形状为 (B, N_img, H, Hc),
            因此我们使用 unsqueeze_dim=2 使 cos/sin 形状变为 (B, N_img, 1, Hc) 
            以便在 H (头数) 维度上广播。
            (Qwen-VL 原始代码默认为 1, 因为它在 (B, H, N_img, Hc) 上操作)
    """
    
    split_sections = mrope_section * 2
    
                            
    cos_chunks = cos.split(split_sections, dim=-1)
    sin_chunks = sin.split(split_sections, dim=-1)
                                                                                                     

                         
    cos_emb = torch.cat(
        [m[i % 3] for i, m in enumerate(cos_chunks)], 
        dim=-1
    ).unsqueeze(unsqueeze_dim)
                                    
    
                         
    sin_emb = torch.cat(
        [m[i % 3] for i, m in enumerate(sin_chunks)], 
        dim=-1
    ).unsqueeze(unsqueeze_dim)
    
             
                                           
    q_embed = (q * cos_emb) + (rotate_half(q) * sin_emb)
    k_embed = (k * cos_emb) + (rotate_half(k) * sin_emb)
    
    return q_embed, k_embed

                                                     

class Multimodal3DRotaryEmbedding(nn.Module):
    """
    Qwen2.5-VL 文本模型使用的 3D RoPE 实现。
    它接收 3D 坐标 (T, H, W) 并为每个坐标计算独立的 RoPE。
    """
    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, theta: float = 10000.0, device=None):
        """
        dim: 应该是 head_dim (Hc)
        """
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        
                                      
                                           
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        x: 任意张量，仅用于获取 device 和 dtype
        position_ids: [3, B, N_img] (T, H, W 坐标)
        """
                                            
                                                                                             
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        
                                           
        position_ids_expanded = position_ids.float().unsqueeze(2)                                                   
                                                                   
        
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):             
                                                                           
                                                     
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
            
                               
            emb = torch.cat((freqs, freqs), dim=-1)
            
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

                                          

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

                                      

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.mrope_section = [8,12,12]                                                              


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos: tuple[torch.Tensor, torch.Tensor], mask=None) -> torch.Tensor:
        """
        x: [B, N_total, C] (N_total = N_images + N_query)
        pos: (cos, sin) 
             cos/sin 形状为 [3, B, N_total, Hc]
        """
        B, N_total, C = x.shape
        qkv = self.qkv(x).reshape(B, N_total, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
                                      
        cos, sin = pos
        
                                                 
                            
        q, k = apply_multimodal_rotary_pos_emb(
            q, 
            k, 
            cos=cos, 
            sin=sin, 
            mrope_section=self.mrope_section,
            unsqueeze_dim=2
        )
        
                                    
        
                               
        
                            
        q = q.transpose(1, 2)                     
        k = k.transpose(1, 2)                     
        v = v.transpose(1, 2)                     


        
                                             
                                                              
                                                                                                                                                           
                                                          
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop)   
        
        x = x.transpose(1, 2).reshape(B, N_total, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
                                   
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, int(hidden_size * mlp_ratio))

    def forward(self, x, pos, mask=None):
                                 
        residual = x
        x = self.norm1(x)
        x = self.attn(x, pos, mask=mask)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x

class QwenImageExtractor(nn.Module):
    def __init__(self, hidden_size=1024,num_layers=16,num_heads=16,query_length=256,patch_size=2,in_chans=16,output_dim=3584):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads   
        self.output_dim=output_dim

        self.patch_embed = PatchEmbed(
            img_size      = None,
            patch_size    = patch_size,
            in_chans      = in_chans,
            embed_dim     = hidden_size,
            strict_img_size=False
        )
        self.query_length=query_length
        self.query = nn.Parameter(torch.randn(1, self.query_length, self.hidden_size))
        
        
                              
        self.blocks = nn.ModuleList([Block(hidden_size, num_heads) for _ in range(self.num_layers)])
        self.norm2 = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

                                     
                           
                                           
        self.rotary_pos_emb = Multimodal3DRotaryEmbedding(self.head_dim)

                             
                                         
        self.cached_pos_ids_shape = None
        self.cached_pos_ids = None

    def _compute_dynamic_pos_ids(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        ⭐️ (修改点 4)
        根据输入的 H 和 W 动态计算 3D RoPE 的 Position IDs
        返回: [3, L_total] (L_images + L_query)
        """
        patch_size_h, patch_size_w = self.patch_embed.patch_size
        grid_h = H // patch_size_h
        grid_w = W // patch_size_w
        
        current_shape = (grid_h, grid_w)

                 
        if current_shape == self.cached_pos_ids_shape:
            if self.cached_pos_ids.device == device:
                return self.cached_pos_ids
            else:
                         
                self.cached_pos_ids = self.cached_pos_ids.to(device)
                return self.cached_pos_ids

                            
        
        num_patches_per_image = grid_h * grid_w

                              
        hpos_ids = torch.arange(grid_h, device=device).unsqueeze(1).expand(-1, grid_w).flatten()
        wpos_ids = torch.arange(grid_w, device=device).unsqueeze(0).expand(grid_h, -1).flatten()
        
                              
                          
        tpos_ids_0 = torch.zeros(num_patches_per_image, device=device, dtype=torch.long)
                        
        pos_ids_0 = torch.stack([tpos_ids_0, hpos_ids, wpos_ids], dim=0)

                              
                          
        tpos_ids_1 = torch.ones(num_patches_per_image, device=device, dtype=torch.long)
                        
        pos_ids_1 = torch.stack([tpos_ids_1, hpos_ids, wpos_ids], dim=0)
        
                           
                            
        img_pos_ids = torch.cat([pos_ids_0, pos_ids_1], dim=1)

                                 
                                                      
        start_idx = max(grid_h, grid_w)
        query_indices = torch.arange(start_idx, start_idx + self.query_length, device=device, dtype=torch.long)
                           
        query_pos_ids = query_indices.unsqueeze(0).expand(3, -1)
        
        all_pos_ids = torch.cat([img_pos_ids, query_pos_ids], dim=1)

                 
        self.cached_pos_ids_shape = current_shape
        self.cached_pos_ids = all_pos_ids
        
        return all_pos_ids

    def forward(self, x):
        """
        x: [B, 2, C, H, W]
        """
        B, N, C, H, W = x.shape
        assert N == 2, "This model is hardcoded for N=2"

                                       
                                  
                                  
        pos_ids = self._compute_dynamic_pos_ids(H, W, device=x.device)
        
                       
                                         
        pos_ids_batch = pos_ids.unsqueeze(1).expand(-1, B, -1)

                                
                                              
        pos_tuple = self.rotary_pos_emb(x, pos_ids_batch)
                               

        x = rearrange(x, "B N C H W -> (B N) C H W")
        x = self.patch_embed(x)
        x = rearrange(x, "(B N) l d -> B (N l) d", B=B, N=N)      
                            
                                   
                                              
        
        
                                                        
                                    
        x = torch.cat([x, self.query.repeat(B, 1, 1)], dim=1)
        
                                    
        if x.shape[1] != pos_ids.shape[1]:
            raise ValueError(
                f"Total sequence length mismatch. "
                f"Input tensor 'x' length is {x.shape[1]}, "
                f"but calculated 3D RoPE length is {pos_ids.shape[1]}. "
                f"Check H/W ({H}/{W}) vs patch_size ({self.patch_embed.patch_size})."
            )

        for block in self.blocks:
                                           
            x = block(x, pos=pos_tuple)  
        x=x[:, -self.query_length:, :]
            
        x = self.norm2(x)
        x = self.output_proj(x)

        return x
    
         
                            
                  
               
                                                  
                                 
                         

                        
              
                                         
                                 
                                                                                                   