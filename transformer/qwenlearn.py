"""
Complete Qwen2-0.5B Implementation
Hand-written PyTorch implementation compatible with official weights
"""

import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer
from safetensors import safe_open

MODEL_DIR = "./data/qwen2-0.5b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)


# Implementation of Qwen2RMSNorm layer
class Qwen2RMSNorm(nn.Module):
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


# Implementation of Qwen2RotaryEmbedding (RoPE) - Fixed
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate the inverse frequency
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, position_ids=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        if position_ids is None:
            # If no position_ids, use default sequential positions
            cos = self.cos_cached[:seq_len].to(dtype=x.dtype)
            sin = self.sin_cached[:seq_len].to(dtype=x.dtype)
        else:
            # position_ids: [batch_size, seq_len] -> flatten and index
            cos = self.cos_cached[position_ids].to(
                dtype=x.dtype
            )  # [batch_size, seq_len, head_dim]
            sin = self.sin_cached[position_ids].to(
                dtype=x.dtype
            )  # [batch_size, seq_len, head_dim]
            # Take the first batch element (they should all be the same for simple sequential ids)
            cos = cos[0]  # [seq_len, head_dim]
            sin = sin[0]  # [seq_len, head_dim]

        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies Rotary Position Embedding to the query and key tensors."""
    # cos, sin: [seq_len, head_dim]
    # q: [batch_size, num_heads, seq_len, head_dim]
    # k: [batch_size, num_kv_heads, seq_len, head_dim]  # Note: different num_heads!

    # Broadcast cos, sin to match q and k shapes separately
    # For q: [batch_size, num_heads, seq_len, head_dim]
    cos_q = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin_q = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]

    # For k: [batch_size, num_kv_heads, seq_len, head_dim]
    cos_k = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin_k = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]

    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed


# Implementation of Qwen2Attention (with Grouped Query Attention - GQA)
class Qwen2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]  # 896
        self.num_heads = config["num_attention_heads"]  # 14
        self.head_dim = self.hidden_size // self.num_heads  # 64
        self.num_key_value_heads = config["num_key_value_heads"]  # 2 (GQA)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 7
        self.max_position_embeddings = config["max_position_embeddings"]
        self.rope_theta = config.get("rope_theta", 10000.0)

        # Validate configuration
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            )

        # Linear projections (shapes verified against official weights)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # RoPE
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Apply linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Handle past key values for generation (if needed)
        if past_key_value is not None:
            # Concatenate past and current key/value
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Update past key value for next generation step
        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        # Expand keys and values for grouped query attention
        # key_states: [bsz, num_key_value_heads, seq_len, head_dim] -> [bsz, num_heads, seq_len, head_dim]
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            causal_mask = attention_mask
            if causal_mask.size() != (bsz, 1, q_len, key_states.shape[-2]):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, key_states.shape[-2])}, but is {causal_mask.size()}"
                )
            attn_weights = attn_weights + causal_mask

        # Apply softmax
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and combine heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Final linear projection
        attn_output = self.o_proj(attn_output)

        if not use_cache:
            past_key_value = None

        return attn_output, attn_weights, past_key_value


# Implementation of Qwen2MLP (SwiGLU activation)
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]  # 896
        self.intermediate_size = config["intermediate_size"]  # 4864

        # SwiGLU components (no bias as verified from official weights)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # SiLU activation function (same as Swish)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # SwiGLU: x -> gate_proj(x) * silu(up_proj(x)) -> down_proj
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Apply SiLU to up projection and multiply with gate
        intermediate = self.act_fn(up) * gate

        # Final down projection
        output = self.down_proj(intermediate)

        return output


# Implementation of Qwen2DecoderLayer
class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]

        # Self attention
        self.self_attn = Qwen2Attention(config)

        # MLP
        self.mlp = Qwen2MLP(config)

        # Layer norms (RMSNorm)
        self.input_layernorm = Qwen2RMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs,
    ):
        # Store residual
        residual = hidden_states

        # Pre-attention layer norm
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )

        # Residual connection after attention
        hidden_states = residual + hidden_states

        # Store residual for MLP
        residual = hidden_states

        # Pre-MLP layer norm
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Residual connection after MLP
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# Implementation of Qwen2Model (main transformer model)
class Qwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])

        # Create the stack of decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config["num_hidden_layers"])
            ]
        )

        # Final layer norm
        self.norm = Qwen2RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        else:
            raise ValueError("You have to specify input_ids")

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Initialize past key values if needed
        if use_cache:
            if past_key_values is None:
                past_key_values = [None] * len(self.layers)
        else:
            past_key_values = [None] * len(self.layers)

        # Pass through all decoder layers
        next_decoder_cache = () if use_cache else None

        for i, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i],
                use_cache=use_cache,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        # Final layer normalization
        hidden_states = self.norm(hidden_states)

        # Prepare outputs
        if use_cache:
            return hidden_states, next_decoder_cache
        else:
            return hidden_states


# Implementation of Qwen2ForCausalLM (Language Modeling Head)
class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.vocab_size = config["vocab_size"]

        # Language modeling head (output projection to vocabulary)
        # Note: According to config, tie_word_embeddings=true, but we implement separate lm_head for clarity
        self.lm_head = nn.Linear(
            config["hidden_size"], config["vocab_size"], bias=False
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        **kwargs,
    ):
        # Forward through the transformer model
        if use_cache:
            transformer_outputs, past_key_values = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            transformer_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = transformer_outputs

        # Project to vocabulary space
        if self.config.get("tie_word_embeddings", False):
            # Use the embedding weights as the output projection (tied weights)
            lm_logits = nn.functional.linear(
                hidden_states, self.model.embed_tokens.weight
            )
        else:
            # Use separate lm_head weights
            lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        output = {
            "logits": lm_logits,
            "past_key_values": past_key_values if use_cache else None,
            "hidden_states": hidden_states,
            "loss": loss,
        }

        return output

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if getattr(self, "_reorder_cache_if_needed", False):
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }


# Weight Loading Function
def load_qwen2_weights(model, weights_path):
    """
    Load weights from official Qwen2 safetensors file into our custom model.

    Args:
        model: Our Qwen2ForCausalLM model instance
        weights_path: Path to the model.safetensors file
    """
    print(f"Loading weights from {weights_path}...")

    # Create weight mapping from official keys to our model keys
    weight_map = {}

    with safe_open(weights_path, framework="pt", device="cpu") as f:
        # Load and map each weight tensor
        for key in f.keys():
            tensor = f.get_tensor(key)

            # Map official weight keys to our model structure
            if key.startswith("model.embed_tokens.weight"):
                our_key = "model.embed_tokens.weight"
            elif key.startswith("model.layers."):
                # Extract layer number and component
                parts = key.split(".")
                layer_idx = parts[2]
                component = ".".join(parts[3:])
                our_key = f"model.layers.{layer_idx}.{component}"
            elif key == "model.norm.weight":
                our_key = "model.norm.weight"
            elif key == "lm_head.weight":
                our_key = "lm_head.weight"
            else:
                print(f"Warning: Unrecognized key {key}")
                continue

            weight_map[our_key] = tensor

    # Load weights into our model
    model_state_dict = model.state_dict()
    loaded_keys = set()

    for our_key, tensor in weight_map.items():
        if our_key in model_state_dict:
            if model_state_dict[our_key].shape == tensor.shape:
                model_state_dict[our_key].copy_(tensor)
                loaded_keys.add(our_key)
            else:
                print(
                    f"❌ Shape mismatch for {our_key}: expected {model_state_dict[our_key].shape}, got {tensor.shape}"
                )
        else:
            print(f"❌ Key not found in model: {our_key}")

    # Check for missing keys
    missing_keys = set(model_state_dict.keys()) - loaded_keys
    if missing_keys:
        print(f"Missing keys: {missing_keys}")

    print(f"Successfully loaded {len(loaded_keys)}/{len(model_state_dict)} weights")

    return model


# Text Generation Implementation
def generate_text(
    model,
    tokenizer,
    input_text,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    pad_token_id=None,
):
    """
    Generate text using our Qwen2 model.

    Args:
        model: Our Qwen2ForCausalLM model
        tokenizer: HuggingFace tokenizer
        input_text: Input text or chat messages
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling
        top_p: Nucleus sampling
        do_sample: Whether to sample or use greedy decoding
        pad_token_id: Padding token ID
    """
    model.eval()

    # Tokenize input
    if isinstance(input_text, str):
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
    else:
        # Assume it's already tokenized
        input_ids = input_text

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    batch_size, input_length = input_ids.shape

    # Initialize generation variables
    past_key_values = None
    generated_tokens = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Prepare input for this step
            if step == 0:
                # First step: use full input
                current_input = input_ids
            else:
                # Subsequent steps: use only the last generated token
                current_input = torch.tensor(
                    [[generated_tokens[-1]]], device=input_ids.device
                )

            # Forward pass
            outputs = model(
                input_ids=current_input, past_key_values=past_key_values, use_cache=True
            )

            # Get logits for the last token
            logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
            past_key_values = outputs["past_key_values"]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(
                    logits, min(top_k, logits.size(-1))
                )
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(-1, top_k_indices, top_k_logits)

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = False

                # Create mask for original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample next token
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1)

            # Add to generated tokens
            generated_tokens.append(next_token.item())

            # Check for end-of-sequence
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode generated text
    full_sequence = torch.cat(
        [input_ids, torch.tensor([generated_tokens]).to(input_ids.device)], dim=1
    )
    generated_text = tokenizer.decode(full_sequence[0], skip_special_tokens=True)
    new_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "full_text": generated_text,
        "new_text": new_text,
        "input_length": input_length,
        "generated_length": len(generated_tokens),
    }


# Configuration for official Qwen2-0.5B model
QWEN2_0_5B_CONFIG = {
    "vocab_size": 151936,
    "hidden_size": 896,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "max_position_embeddings": 131072,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
}


# Example usage
if __name__ == "__main__":
    print("🚀 Qwen2-0.5B Complete Implementation")
    print("=" * 50)

    # Create model
    model = Qwen2ForCausalLM(QWEN2_0_5B_CONFIG)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters"
    )

    # Load weights (if available)
    try:
        model = load_qwen2_weights(model, f"{MODEL_DIR}/model.safetensors")
        print("✅ Weights loaded successfully!")

        # Test with chat template
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "用两句话解释量子纠缠。"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(f"\nInput: {prompt}")

        # Generate response
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            input_text=prompt,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )

        print(f"Response: {result['new_text']}")
        print(f"Generated {result['generated_length']} tokens")

    except Exception as e:
        print(f"Weight loading or generation failed: {e}")
        print("Make sure the safetensors file is available")
