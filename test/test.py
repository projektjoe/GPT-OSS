import torch
import numpy as np

from transformers import AutoTokenizer
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
from official_implementation import Transformer, sdpa, swiglu as swiglu_theirs
from main import GPTOSS, load_config, linear, RMSNorm, attention_head, softmax, rope, SwiGLU as swiglu_ours

# Try to import the built C++ extensions
try:
    import dtypes.bfloat16 as bf
except ImportError:
    # Fall back to the build directory for development
    import dtypes.build.bfloat16 as bf


VERIFY_LAYER_BY_LAYER = True

# ======================================================================
# Utility functions
# ======================================================================

def compare_tensors(name, theirs_tensor, ours_tensor, atol=0):
    """Compare tensors with tolerance."""
    theirs_list = theirs_tensor.tolist()
    ours_list = ours_tensor.to_list()

    if len(theirs_list) != len(ours_list):
        raise ValueError(f"{name}: size mismatch {len(theirs_list)} vs {len(ours_list)}")

    mismatches = 0
    for i, (r, m) in enumerate(zip(theirs_list, ours_list)):
        if not np.allclose(r, m, atol=atol, rtol=0):
            if mismatches < 10:
                print(f"[Mismatch] {name} idx {i}: theirs={r}, ours={m}")
            mismatches += 1

    if mismatches == 0:
        print(f"[OK] {name} passed.")
    else:
        print(f"[FAIL] {name} {mismatches} mismatches.")


def compare_tensors2(name, theirs_tensor, ours_tensor):
    """Strict comparison of tensors (no tolerance)."""
    theirs_list = theirs_tensor.tolist()
    ours_list = ours_tensor.to_list()

    mismatches = sum(1 for t, o in zip(theirs_list, ours_list) if t != o)

    if mismatches == 0:
        print(f"[OK] {name} passed.")
    else:
        print(f"[FAIL] {name} {mismatches} mismatches.")

# ======================================================================
# Setup
# ======================================================================

device = torch.device("cpu")

MODEL_NAME = "openai/gpt-oss-20b"
MODEL_PATH = "gpt-oss-20b/original"

# --- Tokenizer + Harmony prompt ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

messages = [Message.from_role_and_content(Role.USER, "Hi")]
conversation = Conversation.from_messages(messages)
prompt_tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

print("Prompt tokens:", prompt_tokens)

# --- Load reference model ---
theirs_model = Transformer.from_checkpoint(MODEL_PATH, device=device)
config = load_config(MODEL_PATH)

# Save embedding
state = {"embedding": bf.Array(theirs_model.embedding.weight.detach().float().numpy())}

# Embeddings (batched vs sequential)
input_tensor = torch.tensor(prompt_tokens, device=device).unsqueeze(0)
x_theirs_all = theirs_model.embedding(input_tensor)  # (1, seq_len, hidden_size)

x_ours_all = [state["embedding"][tid] for tid in prompt_tokens]

print(f"Processing {len(prompt_tokens)} tokens...")


# ======================================================================
# Forward pass (step by step)
# ======================================================================

# Initialize KV caches
our_k_cache, our_v_cache = [], []

for token_idx, token_id in enumerate(prompt_tokens):
    print(f"\n=== Processing Token {token_idx} (token_id: {token_id}) ===")

    # Current token embeddings
    x_theirs = x_theirs_all[0, 0:token_idx + 1]
    x_ours = x_ours_all[token_idx]

    our_k_cache.append([])
    our_v_cache.append([])

    # Loop over all layers
    for layer_idx, block in enumerate(theirs_model.block):
        print(f"\n--- Layer {layer_idx}, Token {token_idx} ---")

        # ------------------------------------------------------------------
        # Attention norm
        # ------------------------------------------------------------------
        norm_array = bf.Array(block.attn.norm.scale.detach().float().numpy())
        t_theirs = block.attn.norm(x_theirs)
        t_ours = RMSNorm(x_ours, norm_array)
        compare_tensors2(f"block[{layer_idx}].attn.norm token {token_idx}", t_theirs[token_idx], t_ours)
        if VERIFY_LAYER_BY_LAYER:
            t_ours = bf.Array(t_theirs[token_idx].detach().float().numpy())
        # ------------------------------------------------------------------
        # QKV projection
        # ------------------------------------------------------------------
        attn_qkv_weight = bf.Array(block.attn.qkv.weight.detach().float().numpy())
        attn_qkv_bias = bf.Array(block.attn.qkv.bias.detach().float().numpy())

        qkv_theirs = block.attn.qkv(t_theirs)
        qkv_ours = linear(t_ours, attn_qkv_weight, attn_qkv_bias)

        compare_tensors2(f"qkv layer {layer_idx} token {token_idx}", qkv_theirs[token_idx], qkv_ours)
        if VERIFY_LAYER_BY_LAYER:
            qkv_ours = bf.Array(qkv_theirs[token_idx].detach().float().numpy())
        # ------------------------------------------------------------------
        # RoPE
        # ------------------------------------------------------------------
        q_theirs = qkv_theirs[:, :config.num_attention_heads * config.head_dim].contiguous()
        k_theirs = qkv_theirs[
            :, config.num_attention_heads * config.head_dim:
               (config.num_attention_heads + config.num_key_value_heads) * config.head_dim,
        ].contiguous()
        v_theirs = qkv_theirs[
            :, (config.num_attention_heads + config.num_key_value_heads) * config.head_dim:
               (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim,
        ].contiguous()

        q_theirs = q_theirs.view(-1, config.num_key_value_heads,
                                 config.num_attention_heads // config.num_key_value_heads,
                                 config.head_dim)
        k_theirs = k_theirs.view(-1, config.num_key_value_heads, config.head_dim)
        v_theirs = v_theirs.view(-1, config.num_key_value_heads, config.head_dim)

        q_theirs, k_theirs = block.attn.rope(q_theirs, k_theirs)

        q_ours = qkv_ours[:config.num_attention_heads * config.head_dim]
        k_ours = qkv_ours[
            config.num_attention_heads * config.head_dim:
            config.num_attention_heads * config.head_dim + config.num_key_value_heads * config.head_dim
        ]
        v_ours = qkv_ours[
            config.num_attention_heads * config.head_dim + config.num_key_value_heads * config.head_dim:
        ]

        rope(q_ours, k_ours, token_idx, config)

        compare_tensors2(f"rope q layer {layer_idx} token {token_idx}", q_theirs[token_idx].flatten(), q_ours)
        compare_tensors2(f"rope k layer {layer_idx} token {token_idx}", k_theirs[token_idx].flatten(), k_ours)

        # Update KV cache
        our_k_cache[-1].append(k_ours)
        our_v_cache[-1].append(v_ours)

        # ------------------------------------------------------------------
        # Attention
        # ------------------------------------------------------------------
        attn_sink = bf.Array(block.attn.sinks.detach().float().numpy())

        t_theirs = sdpa(q_theirs, k_theirs, v_theirs, block.attn.sinks,
                        block.attn.sm_scale, block.attn.sliding_window)

        att_out = bf.Array([])
        for head_idx in range(config.num_attention_heads):
            out = attention_head(q_ours, our_k_cache, our_v_cache,
                                 attn_sink, head_idx, layer_idx, token_idx, config)
            att_out = att_out.concat(out, axis=0)

        compare_tensors2(f"att layer {layer_idx} token {token_idx}", t_theirs[token_idx], att_out)
        if VERIFY_LAYER_BY_LAYER:
            att_out = bf.Array(t_theirs[token_idx].detach().float().numpy())  # force ours = theirs
        # ------------------------------------------------------------------
        # Linear + residual
        # ------------------------------------------------------------------
        t_theirs = block.attn.out(t_theirs)
        x_theirs = x_theirs + t_theirs

        out_weight = bf.Array(block.attn.out.weight.detach().float().numpy())
        out_bias = bf.Array(block.attn.out.bias.detach().float().numpy())
        h_ours = linear(att_out, out_weight, out_bias)
        x_ours = x_ours + h_ours

        compare_tensors2(f"linear & residual layer {layer_idx} token {token_idx}", x_theirs[token_idx], x_ours)
        if VERIFY_LAYER_BY_LAYER:
            x_ours = bf.Array(x_theirs[token_idx].detach().float().numpy())  # force ours = theirs
        # ------------------------------------------------------------------
        # MLP (MoE routing, experts, swiglu, residual)
        # ------------------------------------------------------------------
        # --- Theirs ---
        t_theirs = block.mlp.norm(x_theirs)
        g_theirs = block.mlp.gate(t_theirs)
        experts_theirs = torch.topk(g_theirs, k=block.mlp.experts_per_token, dim=-1, sorted=True)
        expert_weights_theirs = torch.nn.functional.softmax(experts_theirs.values, dim=-1)
        expert_indices_theirs = experts_theirs.indices

        # --- Ours ---
        mlp_norm = bf.Array(block.mlp.norm.scale.detach().float().numpy())
        mlp_weight = bf.Array(block.mlp.gate.weight.detach().float().numpy())
        mlp_bias = bf.Array(block.mlp.gate.bias.detach().float().numpy())

        h_ours = RMSNorm(x_ours, mlp_norm)
        gate_ours = linear(h_ours.to_list(), mlp_weight, mlp_bias, out_in_f32=True)

        largest_4_experts = sorted(enumerate(gate_ours), key=lambda x: x[1], reverse=True)[:4]
        largest_4_experts_weights = softmax([exp for _, exp in bf.Array(largest_4_experts)])

        compare_tensors2(f"norm layer {layer_idx} token {token_idx}", t_theirs[token_idx], h_ours)
        compare_tensors2(f"gate layer {layer_idx} token {token_idx}",
                         expert_weights_theirs[token_idx], largest_4_experts_weights)

        # --- MoE computation (ours) ---
        expert_outputs = []
        for expert_idx, _ in largest_4_experts:
            w1_out = bf.Array((
                torch.einsum("ck,k->c", block.mlp.mlp1_weight[expert_idx],
                             torch.tensor(h_ours, dtype=torch.bfloat16)) +
                block.mlp.mlp1_bias[expert_idx]
            ).tolist())
            swiglu_out = swiglu_ours(w1_out, config)
            expert_output = (
                torch.einsum("ck,k->c", block.mlp.mlp2_weight[expert_idx],
                             torch.tensor(swiglu_out).to(torch.bfloat16)) +
                block.mlp.mlp2_bias[expert_idx]
            ).tolist()
            expert_outputs.append(expert_output)

        moe_out = linear(largest_4_experts_weights, bf.Array(expert_outputs).T)

        # --- Theirs (MoE forward) ---
        mlp1_weight = block.mlp.mlp1_weight[expert_indices_theirs, ...]
        mlp1_bias = block.mlp.mlp1_bias[expert_indices_theirs, ...]
        t_theirs = torch.einsum("beck,bk->bec", mlp1_weight, t_theirs) + mlp1_bias
        t_theirs = swiglu_theirs(t_theirs, limit=block.mlp.swiglu_limit)

        mlp2_weight = block.mlp.mlp2_weight[expert_indices_theirs, ...]
        mlp2_bias = block.mlp.mlp2_bias[expert_indices_theirs, ...]
        t_theirs = torch.einsum("beck,bek->bec", mlp2_weight, t_theirs)
        t_theirs += mlp2_bias
        t_theirs = torch.einsum("bec,be->bc", t_theirs, expert_weights_theirs)

        compare_tensors2(f"expert_weights layer {layer_idx} token {token_idx}",
                         expert_weights_theirs[token_idx], largest_4_experts_weights)
        compare_tensors2(f"moe layer {layer_idx} token {token_idx}", t_theirs[token_idx], moe_out)
        # Residual
        x_theirs = x_theirs + t_theirs
        x_ours = x_ours + moe_out
        if VERIFY_LAYER_BY_LAYER:
            x_ours = bf.Array(x_theirs[token_idx].detach().float().numpy()) # force ours = theirs

    print(f"\n=== Token {token_idx} processed through all layers ===")


# ======================================================================
# Final processing (last token only)
# ======================================================================

print("\n=== Final processing (last token) ===")

norm = bf.Array(theirs_model.norm.scale.detach().float().numpy())
unembedding = bf.Array(theirs_model.unembedding.weight.detach().float().numpy())

# Ours
h_ours = RMSNorm(x_ours, norm)
logits_ours = unembedding.matmul(h_ours)

# Theirs
logits_theirs = theirs_model.unembedding(torch.tensor(h_ours.to_list(), dtype=torch.bfloat16))

print("Ours predicted token:", np.argmax(logits_ours.to_list()))
print("Theirs predicted token:", np.argmax(logits_theirs.tolist()))

compare_tensors2("final logits", logits_theirs, logits_ours)
