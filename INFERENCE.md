# Inference with GPT-NeoX

Use [generate.py](generate.py) to generate text from a **native NeoX checkpoint**.
The quickest way to check a checkpoint is to run a short prompt with
`temperature: 0.0` and inspect the completion. Set a positive temperature
when you want to sample. Native generation supports greedy decoding and
multinomial sampling, not beam search.

## Run your first prompt

Run the commands below from the NeoX repository root, in an environment with the
NeoX dependencies, DeepSpeed, and the GPUs needed for your checkpoint. No
training dataset or optimizer state is needed for text generation.

1. Start with the **model config used to train the checkpoint**. Match its
   architecture, tokenizer (including vocabulary and merge files, if applicable),
   and tensor/pipeline partitioning. [125M.yml](configs/125M.yml) is an example,
   not a config that can load arbitrary weights. Replace its example hostfile
   with your cluster's hostfile, or omit it for a local launch.
2. In your site config (for example, a copy of [local_setup.yml](configs/local_setup.yml)),
   set `load` to the checkpoint **root**, the directory containing
   `global_step<iteration>/`, not the step directory or a single shard.
   Make the checkpoint and tokenizer paths visible to every worker.
3. Create `configs/infer.yml` with the following contents. The launcher merges
   configs and **rejects duplicate keys**; keep each setting in just one file.
   Use this file *instead of*, not in addition to, [text_generation.yml](configs/text_generation.yml).

```yaml
text_gen_type: input-file
sample_input_file: prompts.txt
sample_output_file: completions.jsonl
temperature: 0.0
maximum_tokens: 32
minimum_tokens: 0
recompute: false
return_logits: false
```

For a matching 125M checkpoint, create an input file and launch:

```bash
printf '%s\n' 'List three primary colors:' 'The capital of France is' > prompts.txt
python deepy.py generate.py -d configs 125M.yml local_setup.yml infer.yml --iteration 10000
head -n 1 completions.jsonl
```

Replace the model/site filenames and `10000` with your own. An explicit
`--iteration` makes a missing step an error. Without it, the loader asks
DeepSpeed for the latest checkpoint; if the log says `Unable to load checkpoint.`,
do not treat the output as coming from trained weights. Use `-i other-prompts.txt`
and `-o other-results.jsonl` to override the input/output paths without editing YAML.
These flags do not switch an explicitly configured `unconditional` run to file mode.
The output directory must exist; the writer **overwrites** the output file.

## Choose a generation mode

| `text_gen_type` | Input and output |
| --- | --- |
| `input-file` | Reads one prompt per line by default, strips surrounding whitespace, and skips empty prompts. Set `prompt_end` to another delimiter for multiline prompts. Writes one JSON record per prompt, in order. |
| `interactive` | Reads a prompt from the terminal on rank zero and prints the growing completion. Press Enter to start another prompt; send EOF (Ctrl-D on Linux) to exit. |
| `unconditional` | Starts each generation from the tokenizer's end-of-document token. Set `num_samples` to the number of completions and `sample_output_file` for JSONL output. With greedy decoding, repeated empty prompts need not produce different text. |

If you omit `text_gen_type`, NeoX chooses `input-file` when you supply
`sample_input_file` and `unconditional` otherwise. Select `interactive` explicitly.

`precompute` is a separate mode for indexed datasets; it does not take text
prompts, sample tokens, or write completion JSONL. Use the split dataset
paths and set `precompute_model_name` for a predictable output suffix.
The current writer has known limitations: a microbatch of one can lose its
batch dimension, and the final batch can wrap back to the start of the dataset.
Check the output document count and lengths against the input before using
precomputed scores.

## Choose greedy decoding or sampling

**Native generation uses temperature to choose the mode.** At zero (the
default), NeoX takes the highest-scoring eligible token, consumes no sampling
randomness, and warns if active sampling filters will be ignored. A filter
cannot turn sampling on. At a positive temperature, NeoX divides the logits by
temperature, applies the active filters, and samples a token. The shipped
`text_generation.yml` example sets temperature to **1.0**, so it samples.

The CLI checks sampling ranges and the minimum/maximum token budget at config
load, before loading the checkpoint. Native generation reuses those settings;
it does not validate again inside the decode loop. A direct Python call or
lm-eval request can supply different settings, so each checks its own options
before its first prompt. When sampling, NeoX caps `top_k` once it knows the
padded vocabulary width.

For example, to sample from a nucleus while enforcing a minimum output length,
edit the values in your generation config:

```yaml
temperature: 0.8
top_p: 0.9
minimum_tokens: 8
maximum_tokens: 64
```

The table is in **application order**, following the
[Transformers v5.14.0 processor order](https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/generation/utils.py#L1273).
Each filter sees the scores left by earlier ones; combining filters is not
equivalent to applying them independently to the raw model probabilities.

| Setting | Accepted values | Effect |
| --- | --- | --- |
| `temperature` | Finite number ≥ 0; default `0` | `0`: greedy. Positive: sample; `1` leaves the scale unchanged. |
| `top_h` | `[0, 1]`; `0` disables | Keeps a high-probability prefix within an entropy budget, considering at most 100 candidates. `1` is **not** the off switch. |
| `top_k` | Nonnegative integer; `0` disables | Keeps the highest-scoring k tokens, including ties. Values above the model's vocabulary width are capped with a warning when sampling. |
| `top_p` | `[0, 1]`; default `1` | Keeps the smallest high-probability prefix reaching the requested mass. An exact match does not keep another token. Both `1` and legacy NeoX `0` disable it. |
| `min_p` | `[0, 1]`; `0` disables | Keeps tokens whose probability is at least `min_p × p(most likely token)`. |
| `typical_p` | `(0, 1]`; `1` disables | Keeps tokens whose surprisal is closest to the distribution's entropy until the requested probability mass is reached. These need not be the most probable tokens. |
| `epsilon_cutoff` | `[0, 1)`; `0` disables | Keeps tokens at or above the probability cutoff, always retaining a highest-probability token. |
| `eta_cutoff` | `[0, 1)`; `0` disables | With `H` the entropy of the current distribution, uses `min(eta_cutoff, sqrt(eta_cutoff) × exp(-H))` as the probability cutoff; always retains a highest-probability token. |

For deterministic output, use zero rather than a tiny positive temperature.
Besides enabling sampling, very small temperatures can make the FP32 entropy
calculation inaccurate for tied logits: for example, `top_h: 0.75` can keep
one of four equally likely tokens at `temperature: 1e-9` instead of three.

## Token budgets and output scores

`minimum_tokens` and `maximum_tokens` count **new tokens**, not the prompt.
Their config defaults are 0 and 64. Both must be nonnegative integers, and
the minimum cannot exceed the maximum. The minimum suppresses EOS (the
end-of-sequence token) for the first N decisions **in both modes**; it does not
disable explicit stop sequences or enlarge the model's context window. A stop
sequence must occur entirely in newly generated tokens for that prompt. When
you pass `stop_tokens` to the native Python API, its completion and token count
include the entire matched sequence, but never EOS. The lm-eval adapter trims
stop strings before returning its response. Leave room for generation: a
prompt using all of `seq_length` has no decode step.

Each file-mode record contains `context` (the prompt), `text` (the completion
only), `length` (its token count), `finished`, `message`, and `duration_seconds`.
Reaching EOS marks a result finished; reaching the token or context cap alone
does not. EOS is not part of the completion. An immediate EOS therefore gives
`text: null`, `length: 0`, and `finished: true`. A zero token budget or
an exactly full prompt instead gives `finished: false` and no score rows.

Set `return_logits: true` only if you need scores: it adds one row of
**processed scores over the full padded vocabulary** for each executed decode
step. They are neither raw model logits nor probabilities. Masked tokens have
score `-inf`; in sampling mode the scores also reflect temperature and filtering.
If a row's highest logit reaches the FP32 magnitude limit under temperature
scaling, NeoX can subtract that logit first. This changes the score offset, not the
sampling probabilities, and preserves ties.
For example, if `minimum_tokens: 2` and EOS wins as soon as it is allowed, the
completion has length **2** but `logits` has **3 rows**. The EOS column is
`-inf` in rows 0 and 1, and row 2 records the EOS decision. There are no rows
after termination. Python's JSON writer represents negative infinity as
`-Infinity`; use a reader that accepts it if you enable scores.

## KV caching, batching, and parallelism

- `recompute: false` (default) reuses the prompt's key/value cache between
  decode steps. `true` runs a full forward pass for every step instead. Use
  `true` for sparse attention or sinusoidal position embeddings: the sinusoidal
  implementation starts positions at zero on each forward. For rotary models
  using the cache, set `rope_fusion: false`; the fused kernel expects its rotary
  table and query to have the same sequence length, which incremental decoding
  does not provide. Alternatively, use `recompute: true`.
- File and list-of-prompts generation serve prompts sequentially. The
  lower-level [stream_tokens API](megatron/text_generation_utils.py) accepts a
  batch of tokenized prompts, but returns scores for **batch item zero only**.
  Supply a nonempty batch and at least one token per row. Unequal-length
  prompts share a budget starting at the shortest prompt; longer prompts get
  fewer new tokens. Exhaust or close the iterator before reusing the model.
  PyTorch gradient recording is disabled in the calling thread while the
  iterator is suspended at a yielded step; finish or close it before doing
  work that requires gradients.
- Use `model_parallel_size` for tensor parallelism and `pipe_parallel_size`
  for pipeline stages. Match the checkpoint layout. Every participating rank
  must enter generation: tensor-parallel scores are gathered before sampling,
  and pipeline stages share the selected token. Only global rank zero writes
  the CLI output file.

The CLI and evaluation adapter configure the model's cache for you. If you
already have a NeoX engine in Python, set its inference mode yourself on
**every participating rank**. For example:

```python
from megatron.text_generation_utils import generate_samples_from_prompt

recompute = neox_args.recompute
model.module.inference_mode(use_cache=not recompute)
records = generate_samples_from_prompt(
    neox_args, model, ["Hello", "Goodbye"], maximum_tokens=16, recompute=recompute
)
```

The prompt owner receives the records; other ranks return an empty list.
An empty **string** is supported here and starts from the tokenizer's EOS token.
In a non-pipeline tensor-parallel job, check tokenized prompt lengths before
launching if you want to truncate or skip oversized inputs. NeoX itself does
not truncate them: when the owner's longest prompt exceeds `seq_length`,
every rank in that TP group raises the same `ValueError` before a model
forward. After catching it on every participating rank, you can submit a
shorter prompt using the same engine and process group. Other data-parallel
replicas can process their own requests independently. A prompt exactly as
long as `seq_length` is valid, but leaves no room to generate.

If you reuse a DeepSpeed pipeline engine, exhaust or explicitly close the
token stream on **every pipeline rank** before starting the next batch. NeoX
clears the KV cache and resets DeepSpeed's activation shapes so the next
training or evaluation batch can use the full sequence length. No manual
`reset_activation_shape()` is needed. If you called `inference_mode()` yourself,
call `model.module.train_mode()` before resuming training; the lm-eval adapter
does this automatically.

## Evaluation harness

The [lm-eval adapter](eval_tasks/eval_adapter.py) uses a **task request's
`do_sample` flag**, unlike native `generate.py`. A missing or false flag is greedy
even if the task supplies a positive temperature or `top_p: 1.0`. With
`do_sample: true`, temperature defaults to 1.0; explicitly setting it to zero
is an error. For example, these are **task** settings, not NeoX model YAML:

```yaml
generation_kwargs:
  do_sample: true
  temperature: 0.8
  top_p: 0.9
  max_gen_toks: 64
  until: ["\n"]
```

For example, launch the built-in [GSM8K generation task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml)
with the checkpoint's model/site configs:

```bash
python deepy.py eval.py -d configs 125M.yml local_setup.yml --iteration 10000 --eval_tasks gsm8k
```

This uses the installed task's own settings. The adapter currently evaluates
**at most 10 examples per task** regardless of the reported `limit`; this
command is a smoke check, not a full benchmark. Generation settings do not
affect tasks that score existing continuations by log likelihood.

- `max_gen_toks` sets the request's new-token budget (adapter default: 128).
  The native CLI's `maximum_tokens` default of 64 is not the adapter default.
- `until` takes a string or list of strings. The response ends before the
  earliest occurrence of any listed stop string in the decoded completion.
  For example, generating tokens `1, 2, 3` with `until: "23"` returns `"1"`;
  generating just `1, 2` before EOS or the token limit returns `"12"`.
  Omitting `until` leaves EOS and the token/context budget as the stopping conditions.
- Requests run sequentially, in order. The adapter only adds greedy results to
  the lm-eval **response cache**; sampled repeats are independent draws, not
  guaranteed different strings. This cache is separate from the model's KV cache.
