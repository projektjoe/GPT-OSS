import os
import json
import numpy as np
import struct
from dataclasses import dataclass
import math

from transformers import AutoTokenizer

from load import load_checkpoint

# Try to import the built C++ extensions
import dtypes.bfloat16 as bf

MATCH_OFFICIAL_IMPL_NUMERICALLY = True

try:
    import torch
except ImportError:
    assert MATCH_OFFICIAL_IMPL_NUMERICALLY == False, "Torch is not installed, cannot replicate the official implementation numerically"

# Try to import torch-based linear layer if available
try:
    from dtypes.linear_layer_torch import linear_torch as linear_torch_
    HAS_TORCH_LINEAR = True
except ImportError:
    HAS_TORCH_LINEAR = False
    linear_torch_ = None  # Define it as None for clarity


# RAM memory requirement warning
import psutil
total_ram_gb = psutil.virtual_memory().total / (1024**3)
if total_ram_gb < 40:
    print(f"⚠️  WARNING: System has {total_ram_gb:.2f} GB RAM (less than 40 GB)")
else:
    print(f"✓ System has {total_ram_gb:.2f} GB RAM")

EOS_TOKEN = 200002



@dataclass
class ModelConfig:
    num_hidden_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    
    
#======================== Helper functions  ==============================#

def load_config(model_path):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        json_config = json.load(f)
        config = ModelConfig(**json_config)
    return config



def f32(x):
    """Round Python float to IEEE-754 float32 and return as Python float."""
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]

def bf16(x):
    # pack to float32 bits
    f32_bits = struct.unpack('<I', struct.pack('<f', float(x)))[0]
    # round-to-nearest-even on the low 16 bits
    lsb = (f32_bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    bf16_bits = (f32_bits + rounding_bias) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', bf16_bits))[0]


#======================== NN functions  ==============================#
def linear(x, W, bias=None, out_in_f32=False):
    m = len(W)
    if m == 0:
        return []
    
    n = len(W[0])

    # sanity checks
    if any(len(row) != n for row in W):
        raise ValueError("W must be rectangular")
    if len(x) != n:
        raise ValueError("x must have length equal to n (inner dims must match)")
    if bias is not None and len(bias) != m:
        raise ValueError("bias must have length equal to m (outer dim of W)")


    out = []
    if bias is not None:
        for i in range(m):
            acc = f32(0.0)
            for k in range(n):
                a = W[i][k]
                b = x[k]
                acc += f32(a * b)
            acc = f32(acc + f32(bias[i]))
            out.append(acc)
    else:
        for i in range(m):
            acc = f32(0.0)
            for k in range(n):
                a = W[i][k]
                b = x[k]
                acc += f32(a * b)
            out.append(acc) 
    if out_in_f32:
        return out
    return bf.Array(out)

def linear(x, W, b=None, out_in_f32= False):
    # Use torch-based linear if available, otherwise fall back to oneDNN
    if HAS_TORCH_LINEAR:
        x_np = np.array(x, dtype=np.float32)
        W_np = np.array(W, dtype=np.float32)
        if b is not None:
            b_np = np.array(b, dtype=np.float32)
        else:
            b_np = np.array([0]*W.shape[0], dtype=np.float32)
        
        if out_in_f32:
            return linear_torch_(x_np, W_np, b_np).tolist()
        return bf.Array(linear_torch_(x_np, W_np, b_np))
    else:
        # Fallback to oneDNN-based linear layer
        from dtypes.linear_layer import linear as linear_dnn
        
        x_np = np.array([x], dtype=np.float32)
        W_np = np.array(W, dtype=np.float32)
        if b is not None:
            b_np = np.array(b, dtype=np.float32)
        else:
            b_np = np.array([0]*W.shape[0], dtype=np.float32)
        if out_in_f32:
            return linear_dnn(x_np, W_np, b_np)[0]
        return bf.Array(linear_dnn(x_np, W_np, b_np))[0]


def softmax(arr):
    max_val = max(arr)
    exps = [math.exp(f32(val) - max_val) for val in arr]
    sum_exps = sum(exps)
    return bf.Array([val / sum_exps for val in exps])


def RMSNorm(x, weights, epsilon=1e-5):
    avg = sum(f32(item) ** 2 for item in x) / len(x)
    inv_sqrt = 1.0 / math.sqrt(avg + epsilon)
    out = [f32(x[i] * inv_sqrt) * f32(weights[i]) for i in range(len(x))]
    return bf.Array(out)


def SwiGLU(x, config, alpha=1.702):
    result = bf.Array([0] * config.hidden_size)
    
    for i in range(config.intermediate_size):
        x_glu = (x[i*2])
        x_linear = (x[i*2 + 1])

        # Clamp the input values
        x_glu = min(config.swiglu_limit, x_glu)
        x_linear = min(config.swiglu_limit, max(-config.swiglu_limit, x_linear))

        gate = (x_glu * bf16(1 / (1 + (math.exp(bf16(-f32(alpha) * x_glu)))))) # SiLU
        result[i] = bf16(gate) * bf16(x_linear + 1)
    return result


def rope(q, k, pos, config):
    if config.rope_scaling_factor > 1.0:
        d_half = config.head_dim / 2

        # Calculate the dimension boundaries for interpolation/extrapolation
        # These are constant and don't need to be in the loop
        low = (
            d_half * math.log(config.initial_context_length / (config.rope_ntk_beta * 2 * math.pi))
            / math.log(config.rope_theta)
        )
        high = (
            d_half * math.log(config.initial_context_length / (config.rope_ntk_alpha * 2 * math.pi))
            / math.log(config.rope_theta)
        )
    
    half_dim = config.head_dim // 2
    
    for h in range(config.num_attention_heads):
        for i in range(0, half_dim): 
            # NTK (YaRN Paper) 
            freq = f32(math.pow(config.rope_theta, (2 * i) / config.head_dim))

            if config.rope_scaling_factor > 1.0:
                # The index for pairs (0, half_dim), (1, half_dim+1), (2, half_dim+2)... is 0, 1, 2...
                # Calculate ramp and clamp it between 0 and 1
                ramp = f32((i - low) / (high - low))
                mask = f32(1 - max(0.0, min(1.0, ramp)))
                
                interpolation = f32(1.0 / (config.rope_scaling_factor * freq))
                extrapolation = f32(1.0 / freq)
                
                # High-frequency dimensions (ramp=1) are interpolated (scaled down).
                # Low-frequency dimensions (ramp=0) are extrapolated (left alone).
                inv_freq = f32(interpolation * (1 - mask) + extrapolation * mask)
            else:
                # If no scaling, theta remains the same
                inv_freq = f32(1.0 / freq)
                
            t = 0.1 * math.log(config.rope_scaling_factor) + 1 if config.rope_scaling_factor > 1.0 else 1.0
            # rope
            angle = f32(pos * inv_freq)
            co, si = bf16(math.cos(angle) * t), bf16(math.sin(angle) * t)

            # Index for real part and imaginary part (separated by half_dim)
            j_real = h * config.head_dim + i
            j_imag = h * config.head_dim + i + half_dim
            
            q_real = f32(q[j_real])
            q_imag = f32(q[j_imag])
            q[j_real] = bf16(q_real * co) - bf16(q_imag * si)
            q[j_imag] = bf16(q_real * si) + bf16(q_imag * co)
            
            if h < config.num_key_value_heads:
                k_real = f32(k[j_real])
                k_imag = f32(k[j_imag])
                k[j_real] = bf16(k_real * co) - bf16(k_imag * si)
                k[j_imag] = bf16(k_real * si) + bf16(k_imag * co)

def attention_head(q, k_cache, v_cache, sinks, head_idx, layer_idx, pos, config):
    sliding_window = layer_idx % 2 == 0
    start = max(0, pos - config.sliding_window) if sliding_window else 0
    
    current_kv_head = head_idx * config.num_key_value_heads // config.num_attention_heads
    
    activations = [0] * (pos + 1)
    
    for i in range(start, pos + 1): 
        score = 0.0
        for j in range(config.head_dim):
            q_idx = (head_idx * config.head_dim) + j
            kv_idx = (config.head_dim * current_kv_head) + j
            score += f32(q[q_idx]) * f32(k_cache[i][layer_idx][kv_idx])
    
        score *= (1.0 / math.sqrt(config.head_dim))
        activations[i] = (score)
    
    activations = bf.Array(activations).concat(bf.Array([sinks[head_idx]]))
    activations = softmax(activations)
    
    out = [0] * config.head_dim
    for i in range(start, pos + 1):
        attention_weight = activations[i]
        for j in range(config.head_dim):
            kv_idx = (config.head_dim * current_kv_head) + j
            v_val = f32(v_cache[i][layer_idx][kv_idx])
            out[j] += f32(attention_weight * v_val)
    
    return bf.Array([(val) for val in out])



#======================== Forward Pass ==============================#

class GPTOSS:
    def __init__(self, config: ModelConfig, checkpoint: dict):
        self.config = config
        self.init_weights(checkpoint)
        # init cache
        self.k_cache = [] # (token, layer, no_of_heads*head_dim)
        self.v_cache = []
    
    def __call__(self, token: int, pos: int):
        self.k_cache.append([])
        self.v_cache.append([])
        
        x = self.embedding[token] # (hidden_size, )

        for l in range(self.config.num_hidden_layers): #TODO rename to l
            
            h = RMSNorm(x, self.attn_norm[l]) # (hidden_size,)
            
            qkv = linear(h, self.attn_qkv_weight[l], self.attn_qkv_bias[l]) # (5120,) =  (5120, 2880) @ (2880,)

            
            q = qkv[:(self.config.num_attention_heads * self.config.head_dim)] # (4096,) = (0:64*64)
            k = qkv[(self.config.num_attention_heads * self.config.head_dim) : 
                    (self.config.num_attention_heads * self.config.head_dim) + (self.config.num_key_value_heads * self.config.head_dim)] # (512,) <= (4096:4608) 
            v = qkv[(self.config.num_attention_heads * self.config.head_dim) + (self.config.num_key_value_heads * self.config.head_dim): ] #(512,) <= (4608:)
                    #(self.config.num_attention_heads + 2* self.config.num_key_value_heads) * self.config.head_dim)]
            

            rope(q, k, pos, self.config)
            
            self.k_cache[-1].append(k)
            self.v_cache[-1].append(v)

            att_out = bf.Array([])
            
            for head_indx in range(self.config.num_attention_heads):
                out = attention_head(q, self.k_cache, self.v_cache, self.attn_sinks[l], head_indx, l, pos, self.config)
                att_out = att_out.concat(out, axis=0)
                
            h = linear(att_out, self.attn_out_weight[l], self.attn_out_bias[l]) 
            
            x = x + h # Residual Layer
            
            # MoE
            h = RMSNorm(x, self.mlp_norm[l]) # (hidden_size,)
            gate = linear(h.to_list(), self.mlp_gate_weight[l], self.mlp_gate_bias[l], out_in_f32=True) # (32,) = (32,2880) x (2880,)
            largest_4_experts = sorted(enumerate(gate), key=lambda x: x[1], reverse=True)[:4]
            largest_4_experts_weights = softmax([expert_logit for _, expert_logit in largest_4_experts]) # activations are percentages/weights of which experts to activate


            moe_out = [0]*self.config.hidden_size # (hidden_size, )

            expert_outputs = []

            if MATCH_OFFICIAL_IMPL_NUMERICALLY:
                
                # use torch's function to match the official implementation numerically.
                for j, (expert_indx, _) in enumerate(largest_4_experts):
                    w1_out = bf.Array(torch.einsum("ck,k->c", 
                                        torch.tensor(self.mlp1_weight[l][expert_indx], dtype=torch.bfloat16),
                                        torch.tensor(h, dtype=torch.bfloat16)).tolist()) + self.mlp1_bias[l][expert_indx]

                    SwiGLU_out = SwiGLU(w1_out, self.config)
                    expert_output = bf.Array(torch.einsum("ck,k->c", 
                                                torch.tensor(self.mlp2_weight[l][expert_indx], dtype=torch.bfloat16), 
                                                torch.tensor(SwiGLU_out, dtype=torch.bfloat16)).tolist()) + self.mlp2_bias[l][expert_indx]

                    expert_outputs.append(expert_output.to_list())
            else:
                for j, (expert_indx, _) in enumerate(largest_4_experts):
                    w1_out = linear(h, self.mlp1_weight[l][expert_indx]) + self.mlp1_bias[l][expert_indx]
                    SwiGLU_out = SwiGLU(w1_out, self.config)
                    expert_output = linear(SwiGLU_out, self.mlp2_weight[l][expert_indx]) + self.mlp2_bias[l][expert_indx]
                    expert_outputs.append(expert_output.to_list())
            

            
            moe_out = linear(largest_4_experts_weights, bf.Array(expert_outputs).T)
            moe_out = bf.Array(moe_out)
            x = x + moe_out # residual
        
        
        h = RMSNorm(x, self.norm) # (hidden_size,)
        
        logits = self.unembedding.matmul(h) # (vocab_size,) = (vocab_size, hidden_size) x (hidden_size, )
        
        return logits
            
        

    def init_weights(self, checkpoint: dict):
        self.embedding = checkpoint['embedding.weight'] # (vocab_size, hidden_size) 

        self.attn_norm = [checkpoint[f'block.{i}.attn.norm.scale'] for i in range(self.config.num_hidden_layers)] #(2880,)
        self.attn_qkv_weight = [checkpoint[f'block.{i}.attn.qkv.weight'] for i in range(self.config.num_hidden_layers)] # (5120, 2880)
        self.attn_qkv_bias = [checkpoint[f'block.{i}.attn.qkv.bias'] for i in range(self.config.num_hidden_layers)] # (5120,)
        self.attn_sinks = [checkpoint[f'block.{i}.attn.sinks'] for i in range(self.config.num_hidden_layers)] # (64,)
        self.attn_out_weight = [checkpoint[f'block.{i}.attn.out.weight'] for i in range(self.config.num_hidden_layers)] # (2880, 4096)
        self.attn_out_bias = [checkpoint[f'block.{i}.attn.out.bias'] for i in range(self.config.num_hidden_layers)] # (2880,)

        self.mlp_norm = [checkpoint[f'block.{i}.mlp.norm.scale'] for i in range(self.config.num_hidden_layers)] # (2880,)
        self.mlp_gate_weight = [checkpoint[f'block.{i}.mlp.gate.weight'] for i in range(self.config.num_hidden_layers)] # (32, 2880)
        self.mlp_gate_bias = [checkpoint[f'block.{i}.mlp.gate.bias'] for i in range(self.config.num_hidden_layers)] # (32,)
        self.mlp1_weight = [checkpoint[f'block.{i}.mlp.mlp1_weight'] for i in range(self.config.num_hidden_layers)] # (32, 5760, 2880)
        self.mlp1_bias = [checkpoint[f'block.{i}.mlp.mlp1_bias'] for i in range(self.config.num_hidden_layers)] # (32, 5760)
        self.mlp2_weight = [checkpoint[f'block.{i}.mlp.mlp2_weight'] for i in range(self.config.num_hidden_layers)] # (32, 2880, 2880)
        self.mlp2_bias = [checkpoint[f'block.{i}.mlp.mlp2_bias'] for i in range(self.config.num_hidden_layers)] # (32, 2880)

        self.norm = checkpoint['norm.scale'] # (2880,)

        self.unembedding = checkpoint['unembedding.weight'] # (vocab_size, hidden_size)

def argmax(arr: list):
    current_max = arr[0]
    max_i = 0
    for i, elem in enumerate(arr):
        if elem > current_max:
            current_max = elem
            max_i = i
    return max_i

def sample(logits, temperature):
    if temperature == 0.0:
        return argmax(logits)
    
def generate(gptoss: GPTOSS, tokenizer: AutoTokenizer, prompt_tokens: list[int], max_tokens: int = 100, temperature=0.0):
    all_tokens = list(prompt_tokens)
    pos = 0
    
    while len(all_tokens) - len(prompt_tokens) < max_tokens:
        current_token = all_tokens[pos]
        logits = gptoss(current_token, pos)
        if pos >= len(prompt_tokens) - 1:
            token = sample(logits, temperature=temperature)
            if token == EOS_TOKEN:
                break
            all_tokens.append(token)
            print(tokenizer.decode([token]), end='', flush=True)
        else:
            print(f'Processed input token: "{tokenizer.decode(current_token)}"')

        pos += 1
    
    return all_tokens[len(prompt_tokens):]  # Return just generated tokens



if __name__ == "__main__":
    model_name = "openai/gpt-oss-20b"
    model_path = "gpt-oss-20b/original"
    config = load_config(model_path)
    
    checkpoint = load_checkpoint(model_path, config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    prompt = "Hi"
    prompt_tokens = tokenizer.encode(f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant") # Added additional tokens for openai harmony
    print("Prompt Tokens:")
    print(prompt_tokens)
    

    gptoss = GPTOSS(config, checkpoint)
    answer = generate(gptoss, tokenizer, prompt_tokens)
    