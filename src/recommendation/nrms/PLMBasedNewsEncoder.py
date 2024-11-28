import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .AdditiveAttention import AdditiveAttention


class PLMBasedNewsEncoder(nn.Module):
    def __init__(
        self,
        pretrained: str = "bert-base-uncased",
        multihead_attn_num_heads: int = 16,
        additive_attn_hidden_dim: int = 200,
    ):
        super().__init__()
        self.plm = AutoModel.from_pretrained(pretrained)

        plm_hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=plm_hidden_size, num_heads=multihead_attn_num_heads, batch_first=True
        )
        self.additive_attention = AdditiveAttention(plm_hidden_size, additive_attn_hidden_dim)

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        V = self.plm(input_val).last_hidden_state  # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        multihead_attn_output, _ = self.multihead_attention(
            V, V, V
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        additive_attn_output = self.additive_attention(
            multihead_attn_output
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        output = torch.sum(
            additive_attn_output, dim=1
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]

        return output

import torch  
from torch import nn  
from transformers import AutoConfig, AutoModel  
from .AdditiveAttention import AdditiveAttention  

class EnhancedPLMBasedNewsEncoder(nn.Module):  
    def __init__(  
        self,  
        pretrained: str = "microsoft/deberta-v3-large",  # 使用更强大的预训练模型  
        multihead_attn_num_heads: int = 16,  
        additive_attn_hidden_dim: int = 200,  
        dropout_rate: float = 0.1  
    ):  
        super().__init__()  
        self.plm = AutoModel.from_pretrained(pretrained)  
        plm_hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size  
        
        # 多层特征提取  
        self.feature_extractor = nn.Sequential(  
            nn.Linear(plm_hidden_size, plm_hidden_size),  
            nn.LayerNorm(plm_hidden_size),  
            nn.ReLU(),  
            nn.Dropout(dropout_rate),  
            nn.Linear(plm_hidden_size, plm_hidden_size)  
        )  
        
        # 多头注意力机制  
        self.multihead_attention = nn.ModuleList([  
            nn.MultiheadAttention(  
                embed_dim=plm_hidden_size,  
                num_heads=multihead_attn_num_heads,  
                batch_first=True,  
                dropout=dropout_rate  
            ) for _ in range(2)  # 使用两层注意力  
        ])  
        
        # 加强版加性注意力  
        self.additive_attention = nn.ModuleList([  
            AdditiveAttention(plm_hidden_size, additive_attn_hidden_dim)  
            for _ in range(2)  
        ])  
        
        # 特征融合层  
        self.feature_fusion = nn.Sequential(  
            nn.Linear(plm_hidden_size * 2, plm_hidden_size),  
            nn.LayerNorm(plm_hidden_size),  
            nn.ReLU(),  
            nn.Dropout(dropout_rate)  
        )  
        
        # 上下文感知层  
        self.context_layer = nn.TransformerEncoderLayer(  
            d_model=plm_hidden_size,  
            nhead=multihead_attn_num_heads,  
            dropout=dropout_rate,  
            batch_first=True  
        )  
        
        # 输出层  
        self.output_layer = nn.Sequential(  
            nn.Linear(plm_hidden_size, plm_hidden_size),  
            nn.LayerNorm(plm_hidden_size),  
            nn.Dropout(dropout_rate)  
        )  

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:  
        # 获取基础特征  
        base_features = self.plm(input_val).last_hidden_state  
        
        # 特征提取  
        enhanced_features = self.feature_extractor(base_features)  
        
        # 多层注意力处理  
        attn_outputs = []  
        current_features = enhanced_features  
        
        for multihead_attn, additive_attn in zip(self.multihead_attention, self.additive_attention):  
            # 多头注意力  
            attn_output, _ = multihead_attn(  
                current_features, current_features, current_features  
            )  
            
            # 加性注意力  
            attn_output = additive_attn(attn_output)  
            
            attn_outputs.append(attn_output)  
            current_features = attn_output  
        
        # 融合不同层的注意力输出  
        multi_layer_features = torch.cat([  
            torch.sum(output, dim=1) for output in attn_outputs  
        ], dim=-1)  
        
        # 特征融合  
        fused_features = self.feature_fusion(multi_layer_features)  
        
        # 上下文处理  
        context_aware_features = self.context_layer(  
            fused_features.unsqueeze(1)  
        ).squeeze(1)  
        
        # 输出处理  
        output = self.output_layer(context_aware_features)  
        
        return output  

    def get_news_embedding(self, input_val: torch.Tensor) -> torch.Tensor:  
        """  
        获取新闻嵌入向量的方法  
        """  
        with torch.no_grad():  
            return self.forward(input_val)