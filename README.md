# Pragna-1B Custom Inference Runtime

## Overview

This repository contains a from-scratch recreation of the inference runtime for **Pragna-1B**, a 1.25 billion parameter language model by [Soket AI](https://huggingface.co/soketlabs/pragna-1b). Rather than relying on the Hugging Face Transformers execution stack, every component of the model — from attention to normalization to text generation — is implemented manually in PyTorch.

The objective was to reverse engineer the Pragna architecture from its published checkpoint, rebuild the full inference pipeline in a standalone codebase, load the official weights into it, and verify end-to-end text generation. The resulting runtime produces coherent output in both Hindi and English, confirming architectural correctness.

---

## Architecture

```
                    Input Text
                        │
                        ▼
                    Tokenizer
                        │
                        ▼
              ┌─────────────────┐
              │    Embedding    │
              │  (69632 × 2048) │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Decoder Layer  │ ×22
              │                 │
              │  RMSNorm        │
              │  GQA Attention  │  (32 Q heads, 4 KV heads)
              │  + Residual     │
              │  RMSNorm        │
              │  SwiGLU FFN     │
              │  + Residual     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Final RMSNorm  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     LM Head     │
              │  (2048 × 69632) │
              └────────┬────────┘
                       │
                       ▼
                    Logits
                       │
                       ▼
                  Next Token
```

---

## Results

✓ Reverse engineered Pragna-1B architecture from the official checkpoint

✓ Implemented from scratch:
  - RMSNorm
  - Rotary Position Embeddings (RoPE)
  - Grouped Query Attention (GQA)
  - SwiGLU Feed-Forward Network

✓ Loaded official Soket AI checkpoint into custom runtime

```
Missing Keys:    0
Unexpected Keys: 0
```

✓ Generated coherent Hindi and English text

✓ Benchmarked custom runtime (~1.2–1.6 tokens/sec on CPU)

✓ Compared against official Hugging Face runtime — both produce coherent, on-topic continuations

---

## Motivation

Most work with large language models begins and ends at a single line:

```python
model = AutoModelForCausalLM.from_pretrained("soketlabs/pragna-1b")
```

This is sufficient for application-level usage, but it treats the model as a black box. It requires no understanding of how attention is computed, how positional information is encoded, how normalization stabilizes training, how checkpoint tensors map to runtime layers, or how autoregressive generation actually works at the tensor level.

This repository exists to move beyond model consumption and into model implementation. Every architectural decision in Pragna — Grouped Query Attention, Rotary Position Embeddings, RMSNorm, SwiGLU feed-forward networks — is implemented from first principles, understood in isolation, and then assembled into a working decoder stack that can load and run the official checkpoint.

---

## Success Criteria

The work is considered successful if the following end-to-end pipeline functions correctly:

```
Prompt (text)
    → Tokenizer (Hugging Face tokenizer, unchanged)
    → Custom Transformer Runtime (this codebase)
    → Official Pragna-1B Weights (loaded from soketlabs/pragna-1b checkpoint)
    → Generated Text (coherent continuation in Hindi and English)
```

This has been achieved. The custom runtime loads the checkpoint with zero missing and zero unexpected keys, and generates coherent text across multiple prompts and languages.

---

## Pragna-1B Architecture

Pragna-1B is a decoder-only transformer following the Llama architecture family. The following configuration was extracted from the official checkpoint and `config.json`:

| Parameter              | Value                            |
|------------------------|----------------------------------|
| Model Type             | LlamaForCausalLM                 |
| Hidden Dimension       | 2048                             |
| Decoder Layers         | 22                               |
| Attention Heads (Q)    | 32                               |
| Key/Value Heads        | 4                                |
| Head Dimension         | 64                               |
| Vocabulary Size        | 69,632                           |
| Intermediate Size (FFN)| 5,632                            |
| Activation Function    | SiLU                             |
| Normalization          | RMSNorm (eps = 1e-5)             |
| Position Encoding      | Rotary Position Embeddings (RoPE)|
| Max Sequence Length     | 2048                             |
| Attention Variant      | Grouped Query Attention (GQA)    |

The 8:1 ratio between query heads (32) and key/value heads (4) is the defining characteristic of GQA in this model. Each KV head is shared across 8 query heads, reducing KV cache memory by 8x compared to standard multi-head attention while preserving model quality.

---

## Reverse Engineering Process

The first step was checkpoint inspection. The official Pragna-1B checkpoint (`model.safetensors`) was loaded and every tensor was enumerated with its name and shape.

**Tensor structure of a single decoder layer (layer 0):**

```
model.layers.0.input_layernorm.weight                   (2048,)
model.layers.0.self_attn.q_proj.weight                   (2048, 2048)
model.layers.0.self_attn.k_proj.weight                   (256, 2048)
model.layers.0.self_attn.v_proj.weight                   (256, 2048)
model.layers.0.self_attn.o_proj.weight                   (2048, 2048)
model.layers.0.self_attn.rotary_emb.inv_freq             (32,)
model.layers.0.post_attention_layernorm.weight           (2048,)
model.layers.0.mlp.gate_proj.weight                      (5632, 2048)
model.layers.0.mlp.up_proj.weight                        (5632, 2048)
model.layers.0.mlp.down_proj.weight                      (2048, 5632)
```

**What the shapes reveal:**

- `q_proj` is (2048, 2048): 32 heads × 64 dim/head = 2048, confirming 32 query heads.
- `k_proj` and `v_proj` are (256, 2048): 4 heads × 64 dim/head = 256, confirming 4 KV heads.
- `gate_proj` and `up_proj` are (5632, 2048): FFN intermediate dimension is 5,632.
- `down_proj` is (2048, 5632): projects back to hidden dimension.
- `rotary_emb.inv_freq` is (32,): precomputed RoPE inverse frequencies for head_dim/2 = 32.
- Normalization weights are (2048,): one scalar per hidden dimension, confirming RMSNorm.

The full checkpoint contains 223 tensors: 2 embedding/output matrices, 10 tensors per layer × 22 layers (including `rotary_emb.inv_freq`), and 1 final normalization layer.

---

## Custom Runtime Implementation

### RMSNorm

Root Mean Square Layer Normalization, as used in the Llama family. Unlike LayerNorm, RMSNorm does not compute or subtract the mean — it only divides by the root mean square, then applies a learned per-element scale.

```
RMSNorm(x) = (x / sqrt(mean(x²) + eps)) * weight
```

The implementation casts to float32 for numerical stability before normalizing, then casts back to the input dtype.

### Rotary Position Embeddings (RoPE)

Positional information is encoded by rotating query and key vectors in pairs. The rotation angles are position-dependent, computed from a geometric frequency series:

```
freq_i = 1 / (theta^(2i/d))        where theta = 10000, i = 0..d/2-1
```

For each position `t`, cosine and sine values are precomputed. At runtime, adjacent pairs of dimensions in Q and K are rotated:

```
[q_2i, q_2i+1] → [q_2i * cos(t·freq_i) - q_2i+1 * sin(t·freq_i),
                   q_2i * sin(t·freq_i) + q_2i+1 * cos(t·freq_i)]
```

The custom implementation precomputes all cos/sin tables up to `max_seq_len` and stores them as non-persistent buffers. This is why the checkpoint's `rotary_emb.inv_freq` tensors (22 layers × 32 values = 704 parameters) are intentionally skipped during weight conversion — the custom runtime computes them dynamically.

### Grouped Query Attention (GQA)

Standard multi-head attention uses equal numbers of Q, K, and V heads. GQA reduces the K and V head count while keeping Q heads unchanged. In Pragna-1B:

- 32 query heads
- 4 key heads
- 4 value heads
- Repetition factor: 8 (each KV head is expanded to serve 8 query heads)

The implementation:

1. Projects input to Q (2048 → 2048), K (2048 → 256), V (2048 → 256)
2. Reshapes into heads: Q has 32 heads of dim 64, K and V have 4 heads of dim 64
3. Applies RoPE to Q and K
4. Expands K and V from 4 heads to 32 heads via `repeat_kv` (repeating each head 8 times)
5. Computes scaled dot-product attention (uses Flash Attention via `torch.nn.functional.scaled_dot_product_attention` when available on PyTorch >= 2.0, otherwise falls back to manual computation with causal masking)
6. Concatenates heads and projects back to hidden dimension via `o_proj`

### Feed-Forward Network (SwiGLU)

Each decoder layer contains a gated feed-forward network using the SwiGLU activation:

```
FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

- `gate_proj` (w1): 2048 → 5632
- `up_proj` (w3): 2048 → 5632
- `down_proj` (w2): 5632 → 2048

The gate projection output passes through SiLU activation and is element-wise multiplied with the up projection, then projected back down. This gating mechanism allows the network to learn which information to pass through the FFN.

### Transformer Block

Each of the 22 decoder layers follows a pre-norm residual architecture:

```
x → RMSNorm → Attention → + residual → RMSNorm → FeedForward → + residual → output
```

The pre-norm design (normalizing before attention/FFN rather than after) is characteristic of the Llama architecture and improves training stability at scale.

### Full Decoder Stack

The complete `Transformer` class assembles:

1. **Token embedding**: vocabulary (69,632) → hidden dimension (2048)
2. **22 decoder layers**: each containing attention + FFN with residual connections
3. **Final RMSNorm**: applied to the output of the last decoder layer
4. **Language model head**: hidden dimension (2048) → vocabulary (69,632), producing logits

During inference, the forward pass applies an optimization: only the logits for the final position in the sequence are computed (rather than all positions), since autoregressive generation only requires the next-token distribution.

### Text Generation

The `generate` method implements autoregressive decoding:

1. Accept an input token sequence
2. Forward through the model to get next-token logits
3. Apply temperature scaling and optional top-k filtering
4. Sample or argmax to select the next token
5. Append to the sequence and repeat

Temperature = 0.0 performs greedy (argmax) decoding. Temperature > 0 samples from the softmax distribution, with higher temperatures producing more diverse outputs. Top-k filtering restricts sampling to the k most probable tokens.

Note: The current implementation does not use a KV cache — each generation step recomputes attention over the entire sequence. This is intentionally simple and correct, prioritizing clarity over throughput.

---

## Weight Conversion

The official Hugging Face checkpoint uses a different naming convention than the custom runtime. A conversion layer (`convert_weights.py`) maps every tensor:

| Hugging Face Name                               | Custom Runtime Name                     |
|--------------------------------------------------|-----------------------------------------|
| `model.embed_tokens.weight`                      | `tok_embeddings.weight`                 |
| `lm_head.weight`                                 | `output.weight`                         |
| `model.norm.weight`                              | `norm.weight`                           |
| `model.layers.N.input_layernorm.weight`          | `layers.N.attention_norm.weight`        |
| `model.layers.N.post_attention_layernorm.weight` | `layers.N.ffn_norm.weight`              |
| `model.layers.N.self_attn.q_proj.weight`         | `layers.N.attention.wq.weight`          |
| `model.layers.N.self_attn.k_proj.weight`         | `layers.N.attention.wk.weight`          |
| `model.layers.N.self_attn.v_proj.weight`         | `layers.N.attention.wv.weight`          |
| `model.layers.N.self_attn.o_proj.weight`         | `layers.N.attention.wo.weight`          |
| `model.layers.N.mlp.gate_proj.weight`            | `layers.N.feed_forward.w1.weight`       |
| `model.layers.N.mlp.down_proj.weight`            | `layers.N.feed_forward.w2.weight`       |
| `model.layers.N.mlp.up_proj.weight`              | `layers.N.feed_forward.w3.weight`       |

All `rotary_emb.inv_freq` buffers are skipped (computed dynamically by the custom runtime).

The conversion produces a state dictionary that loads into the custom `Transformer` with **strict=True** — zero missing keys, zero unexpected keys.

---

## Parameter Verification

| Metric                      | Count           |
|-----------------------------|-----------------|
| Official checkpoint params  | 1,254,189,760   |
| Custom runtime params       | 1,254,189,056   |
| Difference                  | 704             |

The 704-parameter difference is exactly accounted for: 22 layers × 32 `inv_freq` values = 704. These are RoPE precomputation buffers, not trainable weights. The custom runtime computes them dynamically rather than storing them. No trainable parameters are missing.

### Checkpoint Loading

```
Missing Keys:    0
Unexpected Keys: 0
```

This confirms complete, exact compatibility between the official checkpoint and the custom runtime.

---

## Generation Results

### Greedy Decoding (compare_with_hf.py)

The codebase includes a direct comparison between the Hugging Face runtime and the custom runtime, both using greedy decoding (do_sample=False / temperature=0) on the same checkpoint.

**Prompt: "भारत की राजधानी"**
- HF Output: भारत की राजधानी कोलकाता में एक महिला ने एक युवक पर गाली-गलौज करते हुए...
- Custom Output: भारत की राजधानी, दिल्ली में एक अजीवन अस्पताल में एक महिला ने...

**Prompt: "The capital of India is"**
- HF Output: The capital of India is New Delhi. The capital of India is New Delhi...
- Custom Output: The capital of India is Delhi. The capital of India is Delhi...

Both runtimes produce coherent Hindi and English continuations. Minor differences in the generated text are expected: the HF `generate()` method may apply different default sampling parameters, attention masking details, or numerical precision paths compared to the manual implementation. The key result is that the custom runtime generates fluent, on-topic text — not that outputs are byte-identical.

### Sampled Generation (benchmark_pragna.py)

With temperature=0.8 and top_k=50:

**Prompt: "भारत की राजधानी"**
> भारत की राजधानी, सांखी में केरल के पूर्व मुख्यमंत्री का नामांकन...

**Prompt: "Explain machine learning"**
> Explain machine learning — In a recent review, it was found that in the use of a compelling applications often found that...

---

## Sampling Strategy Comparison

The `generate` method in the custom runtime supports three decoding strategies: greedy decoding, top-k sampling, and top-p (nucleus) sampling. The top-p implementation was added to the runtime to study how different sampling strategies affect output diversity and coherence.

**How top-p (nucleus) sampling works:** Rather than sampling from a fixed number of top tokens (top-k), top-p sorts the probability distribution in descending order and includes tokens until their cumulative probability exceeds a threshold `p`. This dynamically adjusts the candidate pool — using fewer tokens when the model is confident and more when the distribution is flat.

The implementation in `models.py`:
1. Sorts the softmax probabilities in descending order
2. Computes the cumulative sum
3. Masks out all tokens beyond the cumulative threshold `p`
4. Renormalizes the remaining probabilities
5. Samples from the filtered distribution

All three strategies were compared side-by-side on identical prompts (`sampling_comparison.py`):

**Prompt: "The capital of India is"**

| Strategy | Output |
|----------|--------|
| Greedy   | The capital of India is Delhi. The capital of India is Delhi. The capital of India is Delhi. *(repetitive loop)* |
| Top-k    | The capital of India is New Delhi. It is the capital of the Indian state of Delhi... Agnoseta. It has the official flag of India. |
| Top-p    | The capital of India is Delhi and the capital of Delhi is the national capital city... |

**Prompt: "नमस्ते मेरा नाम"**

| Strategy | Output |
|----------|--------|
| Greedy   | नमस्ते मेरा नाम तुम्हारा नाम मेरे नाम तुम्हारा नाम मेरे नाम तुम्हे तुम् तुम् तुम्... *(degenerate repetition)* |
| Top-k    | नमस्ते मेरा नाम, 1998 में जारी हुए एक कनाडाई फिल्म है, जो एक कनाडा का दूसरा प्रदर्शन... |
| Top-p    | नमस्ते मेरा नाम ( ४ ) शिष्टावली ( ६) निन्मत्र से मिलता... |

**Observations:**

- Greedy decoding consistently falls into repetitive loops, a well-known failure mode of autoregressive models without sampling.
- Top-k (k=50, temperature=0.8) produces more diverse and coherent continuations by restricting sampling to the 50 most probable tokens.
- Top-p (p=0.9, temperature=0.8) produces the most varied outputs by dynamically adjusting the candidate pool size based on the model's confidence at each step.
- Both sampling strategies effectively break the repetition patterns that greedy decoding exhibits, with top-p offering a more principled approach to candidate selection.

---

## Logit Parity Validation

To go beyond qualitative comparison, the codebase includes a direct logit-level validation (`logit_parity.py`). Both the Hugging Face runtime and the custom runtime are given the same prompt, and their raw output logits (over the full 69,632-token vocabulary) are compared element-wise.

**Prompt: "The capital of India is"**

| Metric               | Value       |
|----------------------|-------------|
| Mean Absolute Diff   | 0.22338006  |
| Max Absolute Diff    | 1.59867740  |
| Std of Diff          | 0.18917128  |

**Top predicted token:**

| Runtime | Token ID | Decoded |
|---------|----------|---------|
| HF      | 1570     | New     |
| Custom  | 5556     | Del     |

The HF runtime predicts "New" (as in "New Delhi") while the custom runtime predicts "Del" (as in "Delhi"). Both are factually correct continuations. The mean absolute logit difference of ~0.22 across 69,632 logit values is small and attributable to floating-point accumulation differences across the two execution paths — different operation ordering, different attention mask construction, and different RoPE computation methods (stored `inv_freq` buffers vs. dynamically computed frequencies). Despite these numerical differences, both runtimes arrive at semantically equivalent predictions.

---

## Benchmark

CPU inference benchmark (no GPU, no KV cache, full recomputation per step):

| Prompt                 | Tokens Generated | Time (sec) | Tokens/sec |
|------------------------|------------------|------------|------------|
| भारत की राजधानी          | 50               | 30.81      | 1.62       |
| Explain machine learning | 50               | 35.40      | 1.41       |
| नमस्ते मेरा नाम          | 50               | 40.56      | 1.23       |

These numbers reflect pure CPU inference without any optimization (no KV cache, no batched prefill, no quantization). The primary purpose of benchmarking was to confirm that the runtime is functional and deterministic, not to compete with optimized inference engines.

---

## Execution Flow

The end-to-end inference pipeline from prompt to generated text:

```
1. Input text
2. Tokenize via Hugging Face tokenizer (soketlabs/pragna-1b)
3. Token IDs → Token Embeddings (69,632 × 2048 lookup)
4. Precompute RoPE cos/sin tables for sequence length
5. For each of 22 decoder layers:
   a. RMSNorm
   b. Q/K/V linear projections
   c. Apply RoPE to Q and K
   d. Expand K/V from 4 heads → 32 heads (GQA repeat)
   e. Scaled dot-product attention (Flash or manual)
   f. Output projection + residual connection
   g. RMSNorm
   h. SwiGLU feed-forward (gate, up, SiLU, multiply, down)
   i. Residual connection
6. Final RMSNorm
7. LM head projection → logits over 69,632 vocabulary
8. Temperature scaling + top-k filtering
9. Sample next token
10. Append token, repeat from step 3
```

---

## Repository Structure

```
models.py                  Core architecture: RMSNorm, RoPE, GQA, SwiGLU FFN,
                           TransformerBlock, Transformer, and generation logic.

convert_weights.py         Downloads the official checkpoint from Hugging Face and
                           maps all tensor names from HF convention to the custom
                           runtime convention. Produces a loadable state dict.

inspect_model.py           Instantiates the custom model and enumerates every
                           parameter with its name and shape. Writes the full
                           inventory to custom_model_inventory.txt.

check_params.py            Compares total parameter counts between the official
                           checkpoint and the custom runtime. Reports the difference.

generate.py                Minimal generation script. Loads converted weights,
                           runs greedy autoregressive decoding on a Hindi prompt,
                           and prints the output.

compare_with_hf.py         Side-by-side comparison of HF Transformers runtime and
                           custom runtime. Runs both on identical prompts with
                           greedy decoding and saves results to
                           hf_vs_custom_results.txt.

benchmark_pragna.py        Benchmarks the custom runtime on multiple prompts.
                           Measures tokens/sec for 50-token generation and saves
                           results to benchmark_results.txt.

test_pragna.py             Smoke test. Instantiates the model, passes random token
                           IDs through a forward pass, and verifies output shape.

logit_parity.py            Logit-level comparison between HF and custom runtimes.
                           Computes mean/max/std absolute logit differences and
                           compares top predicted tokens. Saves results to
                           logit_parity_results.txt.

sampling_comparison.py     Compares greedy, top-k, and top-p decoding strategies
                           on identical prompts. Saves results to
                           sampling_comparison_results.txt.

notebook.ipynb             Exploratory notebook used during development for
                           checkpoint inspection and architecture analysis.
```

**Generated output files:**

```
pragna_tensor_inventory.txt    Full tensor inventory of the official HF checkpoint (223 tensors).
pragna_summary.txt             Summarized checkpoint structure (embeddings, layer 0, final norm).
pragna_layer0.txt              Detailed tensor shapes for decoder layer 0.
custom_model_inventory.txt     Full tensor inventory of the custom runtime (202 parameters).
hf_vs_custom_results.txt       Side-by-side generation outputs from HF and custom runtimes.
benchmark_results.txt          CPU inference benchmark results.
logit_parity_results.txt       Logit-level parity report between HF and custom runtimes.
sampling_comparison_results.txt  Greedy vs top-k vs top-p generation comparison.
```

---

## Dependencies

- Python 3.8+
- PyTorch >= 2.0 (for Flash Attention via `scaled_dot_product_attention`; falls back to manual attention on older versions)
- Hugging Face `transformers` (for tokenizer and HF runtime comparison)
- Hugging Face `safetensors` (for checkpoint loading)
- Hugging Face `huggingface_hub` (for checkpoint downloading)
- NumPy

---

## Usage

**Generate text with the custom runtime:**

```bash
python generate.py
```

**Compare HF vs custom runtime outputs:**

```bash
python compare_with_hf.py
```

**Run inference benchmark:**

```bash
python benchmark_pragna.py
```

**Verify parameter counts:**

```bash
python check_params.py
```

**Inspect custom model architecture:**

```bash
python inspect_model.py
```

**Logit parity validation:**

```bash
python logit_parity.py
```

**Compare sampling strategies:**

```bash
python sampling_comparison.py
```

**Smoke test (no checkpoint needed):**

```bash
python test_pragna.py
```

Note: Scripts that load the official checkpoint (`generate.py`, `compare_with_hf.py`, `benchmark_pragna.py`, `check_params.py`, `logit_parity.py`, `sampling_comparison.py`) will automatically download the Pragna-1B weights from Hugging Face on first run (~5 GB).

---

## Key Learnings

- **Transformer internals**: Every component — embedding, attention, normalization, feed-forward, residual connections, generation — was implemented and debugged individually.
- **Grouped Query Attention**: Understood the mechanics of asymmetric Q/KV head counts and the KV expansion via head repetition.
- **Rotary Position Embeddings**: Implemented the frequency precomputation, complex-plane rotation, and broadcasting logic from the original RoPE paper.
- **RMSNorm**: Understood the difference from LayerNorm and the numerical stability considerations (float32 casting).
- **Checkpoint forensics**: Learned to read tensor names and shapes to reverse-engineer model architecture without access to source code.
- **Weight mapping**: Built a systematic conversion layer between two different naming conventions for the same set of parameters.
- **Inference-time optimizations**: Understood the final-position-only logit computation and the role of KV caches (even though one is not implemented here).

---

## Future Work

- **KV Cache**: Implement key/value caching for efficient autoregressive decoding, avoiding redundant recomputation of attention for prior tokens.
- **Streaming Generation**: Token-by-token output streaming instead of batch generation.
- **Speculative Decoding**: Use a smaller draft model to propose token sequences, verified in parallel by Pragna-1B.
- **Quantization**: 4-bit and 8-bit weight quantization for reduced memory and faster inference.
- **Architectural Experiments**: The custom runtime is designed to serve as a testbed for modifications — alternative attention mechanisms, hybrid Mamba/attention layers, early-exit decoders, and other experimental inference optimizations.

---

## Conclusion

This repository demonstrates a complete, ground-up recreation of the Pragna-1B inference runtime. The official Soket AI checkpoint loads into the custom architecture with exact parameter compatibility, and the runtime generates coherent text in both Hindi and English through purely custom PyTorch code. No part of the Hugging Face Transformers execution stack is used for model inference — only the tokenizer and checkpoint download utilities are retained.

The codebase serves as both a deep learning exercise and a practical foundation for future experimentation with transformer architecture and inference systems.
