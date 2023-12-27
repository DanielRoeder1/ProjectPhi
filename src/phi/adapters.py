from torch import nn
import torch

class SharedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj_q_shared = nn.Linear(config.hidden_size*2, config.hidden_size, bias= config.know_proj_bias)
        self.proj_k_know = nn.Linear(config.hidden_size, config.hidden_size, bias= config.know_proj_bias)
        self.proj_k_mlp = nn.Linear(config.hidden_size, config.hidden_size, bias= config.know_proj_bias)
        self.proj_v_know = nn.Linear(config.hidden_size, config.hidden_size, bias= config.know_proj_bias)
        self.proj_v_mlp = nn.Linear(config.hidden_size, config.hidden_size, bias= config.know_proj_bias)
        self.proj_o = nn.Linear(config.hidden_size, config.hidden_size, bias= config.know_proj_bias)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.attn_dropout = nn.Dropout(config.attn_pdrop)

        self.scale_attn_weights = config.scale_attn_weights
        self.embed_dim = config.hidden_size

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )

    def forward(self, hidden_states, knowledge, attention_mask=None):
        shared_states = torch.concat((hidden_states, knowledge), dim = -1)
        query = self.proj_q_shared(shared_states)
        key_know = self.proj_k_know(knowledge)
        key_mlp = self.proj_k_mlp(hidden_states)
        value_know = self.proj_v_know(knowledge)
        value_mlp = self.proj_v_mlp(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key_know = self._split_heads(key_know, self.num_heads, self.head_dim)
        key_mlp = self._split_heads(key_mlp, self.num_heads, self.head_dim)
        value_know = self._split_heads(value_know, self.num_heads, self.head_dim)
        value_mlp = self._split_heads(value_mlp, self.num_heads, self.head_dim)

        scores_know = torch.matmul(query, key_know.transpose(-1, -2))
        scores_know = self.maybe_scale_attn(scores_know)
        scores_know = self.apply_causal_mask(scores_know)

        scores_mlp = torch.matmul(query, key_mlp.transpose(-1, -2))
        scores_mlp = self.maybe_scale_attn(scores_mlp)
        scores_mlp = self.apply_causal_mask(scores_mlp)

        if attention_mask is not None:
            scores_know = scores_know + attention_mask
            scores_mlp = scores_mlp + attention_mask

        concat_scores = torch.cat((scores_know, scores_mlp), dim = -1)
        concat_scores = torch.softmax(concat_scores, dim = -1)
        concat_scores = self.attn_dropout(concat_scores)
        concat_values = torch.cat((value_know, value_mlp), dim = -2)

        out = concat_scores @ concat_values
        out = self._merge_heads(out, self.num_heads, self.head_dim)
        out = self.proj_o(out)

        return out, concat_scores
    
    def apply_causal_mask(self, attn_scores):
        # From GPT2 attention
        query_length, key_length = attn_scores.size(-2), attn_scores.size(-1)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores.to(attn_scores.dtype), mask_value)
        return attn_scores
    
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def maybe_scale_attn(self, attn_scores):
        if self.scale_attn_weights:
            attn_scores = attn_scores / torch.full(
                [], self.embed_dim ** 0.5, dtype=attn_scores.dtype, device=attn_scores.device
            )
        return attn_scores


  

class Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.proj_q = nn.Linear(config.hidden_size, config.hidden_size, bias= config.know_proj_bias)
        self.proj_k = nn.Linear(config.hidden_size, config.hidden_size, bias= config.know_proj_bias)
        self.proj_v = nn.Linear(config.hidden_size, config.hidden_size, bias= config.know_proj_bias)
        self.proj_o = nn.Linear(config.hidden_size, config.hidden_size, bias= config.know_proj_bias)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

    def forward(self, query, key, value, attention_mask=None):
        query = self.proj_q(query)
        key = self.proj_k(key)
        value = self.proj_v(value) 
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.proj_o(attn_output)

        return attn_output, attn_weights
    

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        self.attention = Attention(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        output_attentions,
    ):
        attn_out, attn_weights = self.attention(query = hidden_states, 
                                                key = encoder_hidden_states, 
                                                value = encoder_hidden_states, 
                                                attention_mask = encoder_attention_mask)

        hidden_states = attn_out * self.attn_gate.tanh()

        return hidden_states, attn_weights