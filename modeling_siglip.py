import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid', # no padding
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.register_buffer('position_ids',
                             torch.arange(self.num_positions).expand((-1,1)),
                             persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        _, _, height, width = pixel_values.shape
        # pixel_values: [Batch_Size, num_channels = 3, H, W]
        # -> patch_embeds: [Batch_size, Embed_dim, Num_Patch_H, Num_Patch_W]
        patch_embeds = self.patch_embedding(pixel_values)

        # embeddings: [Batch_size, embed_dim, Num_Patch] Num_Patch = Num_Patch_H * Num_Patch_W
        embeddings = patch_embeds.flatten(2)

        # embeddings: [Batch_size, Num_Patch, embed_dim]
        embeddings = embeddings.transpose(1, 2)

        # Add position embeddings to each patch
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # [Batch_Size, Num_Patchs, Embed_dim] -> [Batch_Size, Num_Patchs, Intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # [Batch_Size, Num_Patchs, Intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        # [Batch_Size, Num_Patchs, Embed_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # Equivalent to 1/sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.FloatTensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # hidden_states: [Batch_size, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_size, num_patches, embed_dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_size, num_patches, embed_dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_size, num_patches, embed_dim]
        value_states = self.v_proj(hidden_states)

        ### We want each head learn a part of each token, not the full token
        ### like in language, a word can be both verb, noun and adj, so a head can be learned as a noun, the other is verb or adj
        # query_states: [Batch_size, Num_head, num_patches, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # attn_weights: [Batch_Size, num_heads, num_patches, num_patches]...
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)} but got {attn_weights.size()}"
            )

        # Apply softmax row-wise, attn_weights: [batch_size, num_heads, Num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Multiply the attention weights by the value states. attn_output: [Batch_Size, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # Concatenate all the head
        # First transpose it back to [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, num_patches, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # then multiply with w0:[embed_dim, embed_dim] to mix the head result together
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=self.config.attention_dropout)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=self.config.attention_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Residual: [Batch_Size, Num_Patchs, Embed_dim]
        residual = hidden_states

        # [Batch_Size, Num_Patchs, Embed_dim] -> [Batch_Size, Num_Patchs, Embed_dim]
        hidden_states = self.layer_norm1(hidden_states)

        # [Batch_Size, Num_Patchs, Embed_dim] -> [Batch_Size, Num_Patchs, Embed_dim]
        hidden_states, _ = self.self_attn(hidden_states)

        # Add residual
        # [Batch_Size, Num_Patchs, Embed_dim] -> [Batch_Size, Num_Patchs, Embed_dim]
        hidden_states = hidden_states + residual
        residual = hidden_states

        # [Batch_Size, Num_Patchs, Embed_dim] -> [Batch_Size, Num_Patchs, Embed_dim]
        hidden_states = self.layer_norm2(hidden_states)

        # [Batch_Size, Num_Patchs, Embed_dim] -> [Batch_Size, Num_Patchs, Embed_dim]
        hidden_states = self.mlp(hidden_states)

        # Add residual
        hidden_states = hidden_states + residual
        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_image_tokens)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # input_embeds: [batch_size, num_pathches, embed_dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
            hidden_states = encoder_layer(hidden_states)
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(
            self,
            config: SiglipVisionConfig,
    ):
        super().__init__()
        self.config = config
        embed_dim = self.config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=self.config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_size, channels, height, width] -> [Batch_Size, num_patchs, Embed_dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_states = self.encoder(inputs_embeds=hidden_states)
        last_hidden_states = self.post_layernorm(last_hidden_states)
        return last_hidden_states

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_value) -> tuple:
        # [Batch_Size, Channels, Height, Weight] => [Batch_size, Num_Patchs, Embed_Dim] Num_Patchs is num_image_tokens
        return self.vision_model(pixel_values=pixel_value)
