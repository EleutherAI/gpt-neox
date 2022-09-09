import json
import yaml
import argparse
import os

BASE_CONFIG = {
  "hidden_act": "gelu",
  "architectures": [
    "GPTNeoXForCausalLM"
  ],
  "bos_token_id": 0,
  "eos_token_id": 0,
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-05,
  "model_type": "gpt_neox",
  "hidden_size": 1024,
  "intermediate_size": 4096,
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "max_position_embeddings": 2048,
  "rotary_pct": 1.0,
  "rotary_emb_base": 10000,
  "torch_dtype": "float16",
  "use_cache": True,
  "vocab_size": 50304,
}

NEOX_HF_MAPS = {
  "hidden_size": "hidden-size",
  "num_attention_heads": "num-attention-heads",
  "num_hidden_layers": "num-layers",
  "max_position_embeddings": "max-position-embeddings",
}

def make_json_file(neox_config, vocab_size, save_dir):
    for hf_key, neox_key in NEOX_HF_MAPS.items():
      neox_val = neox_config.get(neox_key)
      if neox_val is not None:
        BASE_CONFIG[hf_key] = neox_val
    
    rotary_ndims = neox_config.get('rotary_ndims', 64)
    hidden_size = BASE_CONFIG['hidden_size']
    num_heads = BASE_CONFIG['num_attention_heads']

    # hidden_size / num_heads : 128
    estimated_rotary = hidden_size // num_heads
    rotary_pct = rotary_ndims / estimated_rotary
    BASE_CONFIG['rotary_pct'] = rotary_pct

    BASE_CONFIG['intermediate_size'] = hidden_size * 4
    BASE_CONFIG['vocab_size'] = vocab_size

    with open(os.path.join(save_dir, "config.json"),'w') as f:
      json.dump(BASE_CONFIG, f)


def main():
    parser = argparse.ArgumentParser(description='making config.')
    parser.add_argument('--config_file', type=str,
                        help='yml config file path (not relative path), which should be ~.yml')
    parser.add_argument('--vocab_size', type=int, help='set vocab size of models.')
    parser.add_argument('--output_dir', type=str,
                        help='Output dir, to save the 1-GPU weights configs')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
      config = yaml.load(f, Loader=yaml.FullLoader)

    make_json_file(config, args.vocab_size, args.output_dir)


if __name__ == '__main__':
    main()
