# refered from : https://github.com/VHellendoorn/Code-LMs/blob/0034209056ba958268eb0c71f9f0521a3ff4c962/Convert2HF/convert_neox_pt_to_huggingface_neox.py
# -*- coding: utf-8 -*-


import sys
import os
import torch
from collections import OrderedDict
import argparse




def main():
    parser = argparse.ArgumentParser(description='Merge 20B checkpoint.')
    parser.add_argument('--input_dir', type=str,
                        help='Checkpoint dir, which should contain (e.g. a folder named "global_step150000")')
    parser.add_argument('--output_dir', type=str,
                        help='Output dir, to save the 1-GPU weights configs')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = os.path.join(args.output_dir, "pytorch_model.bin")

    layer_files = []
    layer_id = -1
    state_dict = OrderedDict()
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith("layer_"):
                # print(file)
                layer_files.append(os.path.join(root, file))


    layer_files = sorted(layer_files)
    for file in layer_files:
        # print(file)
        new_layer = True

        module = torch.load(file, map_location=torch.device('cpu'))
        print(module.keys())

        for key, value in module.items():
            if "word_embeddings" in key:
                new_key = key.replace("word_embeddings", "gpt_neox.embed_in")
                state_dict[new_key] = value
            elif "_layernorm" in key or "attention" in key or "mlp" in key:
                if new_layer:
                    layer_id += 1
                    new_layer = False
                new_key = "gpt_neox.layers." + str(layer_id) + "." + key
                state_dict[new_key] = value
            elif key.startswith("norm."):
                new_key = "gpt_neox.final_layer_norm." + key.split(".")[-1]
                state_dict[new_key] = value
            elif "final_linear" in key:
                new_key = "embed_out." + key.split(".")[-1]
                state_dict[new_key] = value
            # print("Convert \"{}\" to \"{}\"".format(key, new_key))

    # print(state_dict.keys())
    torch.save(state_dict, output_file)


if __name__ == '__main__':
    main()

# ['gpt_neox.layers.21.attention.bias', 'gpt_neox.layers.16.post_attention_layernorm.bias', 'gpt_neox.layers.3.post_attention_layernorm.bias', 'gpt_neox.layers.19.attention.bias', 'gpt_neox.layers.7.attention.masked_bias', 'gpt_neox.layers.17.attention.bias', 'gpt_neox.layers.1.attention.bias', 'gpt_neox.layers.14.post_attention_layernorm.bias', 'gpt_neox.layers.7.attention.bias', 'gpt_neox.layers.6.post_attention_layernorm.weight', 'gpt_neox.layers.14.post_attention_layernorm.weight', 'gpt_neox.layers.9.post_attention_layernorm.bias', 'gpt_neox.layers.4.attention.bias', 'gpt_neox.layers.2.attention.bias', 'gpt_neox.layers.15.post_attention_layernorm.weight', 'gpt_neox.layers.0.attention.bias', 'gpt_neox.layers.6.attention.bias', 'gpt_neox.layers.2.attention.masked_bias', 'gpt_neox.layers.18.attention.bias', 'gpt_neox.layers.17.post_attention_layernorm.weight', 'gpt_neox.layers.10.post_attention_layernorm.weight', 'gpt_neox.layers.18.attention.masked_bias', 'gpt_neox.layers.11.post_attention_layernorm.weight', 'gpt_neox.layers.7.post_attention_layernorm.weight', 'gpt_neox.layers.2.post_attention_layernorm.bias', 'gpt_neox.layers.9.attention.masked_bias', 'gpt_neox.layers.20.attention.masked_bias', 'gpt_neox.layers.21.post_attention_layernorm.bias', 'gpt_neox.layers.23.attention.masked_bias', 'gpt_neox.layers.12.post_attention_layernorm.bias', 'gpt_neox.layers.0.post_attention_layernorm.weight', 'gpt_neox.layers.11.post_attention_layernorm.bias', 'gpt_neox.layers.9.post_attention_layernorm.weight', 'gpt_neox.layers.20.attention.bias', 'gpt_neox.layers.11.attention.bias', 'gpt_neox.layers.16.post_attention_layernorm.weight', 'gpt_neox.layers.19.post_attention_layernorm.bias', 'gpt_neox.layers.23.attention.bias', 'gpt_neox.layers.10.attention.masked_bias', 'gpt_neox.layers.20.post_attention_layernorm.bias', 'gpt_neox.layers.10.post_attention_layernorm.bias', 'gpt_neox.layers.15.attention.bias', 'gpt_neox.layers.1.post_attention_layernorm.weight', 'gpt_neox.layers.15.attention.masked_bias', 'gpt_neox.layers.16.attention.bias', 'gpt_neox.layers.6.attention.masked_bias', 'gpt_neox.layers.22.attention.masked_bias', 'gpt_neox.layers.2.post_attention_layernorm.weight', 'gpt_neox.layers.3.post_attention_layernorm.weight', 'gpt_neox.layers.15.post_attention_layernorm.bias', 'gpt_neox.layers.22.attention.bias', 'gpt_neox.layers.3.attention.bias', 'gpt_neox.layers.21.attention.masked_bias', 'gpt_neox.layers.8.attention.bias', 'gpt_neox.layers.5.attention.masked_bias', 'gpt_neox.layers.17.attention.masked_bias', 'gpt_neox.layers.4.post_attention_layernorm.bias', 'gpt_neox.layers.12.attention.masked_bias', 'gpt_neox.layers.0.post_attention_layernorm.bias', 'gpt_neox.layers.22.post_attention_layernorm.bias', 'gpt_neox.layers.0.attention.masked_bias', 'gpt_neox.layers.12.post_attention_layernorm.weight', 'gpt_neox.layers.8.post_attention_layernorm.weight', 'gpt_neox.layers.22.post_attention_layernorm.weight', 'gpt_neox.layers.13.attention.masked_bias', 'gpt_neox.layers.17.post_attention_layernorm.bias', 'gpt_neox.layers.5.post_attention_layernorm.weight', 'gpt_neox.layers.20.post_attention_layernorm.weight', 'gpt_neox.layers.13.attention.bias', 'gpt_neox.layers.1.attention.masked_bias', 'gpt_neox.layers.13.post_attention_layernorm.bias', 'gpt_neox.layers.13.post_attention_layernorm.weight', 'gpt_neox.layers.23.post_attention_layernorm.bias', 'gpt_neox.layers.19.post_attention_layernorm.weight', 'gpt_neox.layers.5.attention.bias', 'gpt_neox.layers.23.post_attention_layernorm.weight', 'gpt_neox.layers.9.attention.bias', 'gpt_neox.layers.4.post_attention_layernorm.weight', 'gpt_neox.layers.14.attention.masked_bias', 'gpt_neox.layers.16.attention.masked_bias', 'gpt_neox.layers.18.post_attention_layernorm.bias', 'gpt_neox.layers.19.attention.masked_bias', 'gpt_neox.layers.7.post_attention_layernorm.bias', 'gpt_neox.layers.6.post_attention_layernorm.bias', 'gpt_neox.layers.12.attention.bias', 'gpt_neox.layers.11.attention.masked_bias', 'gpt_neox.layers.5.post_attention_layernorm.bias', 'gpt_neox.layers.8.post_attention_layernorm.bias', 'gpt_neox.layers.18.post_attention_layernorm.weight', 'gpt_neox.layers.10.attention.bias', 'gpt_neox.layers.3.attention.masked_bias', 'gpt_neox.layers.21.post_attention_layernorm.weight', 'gpt_neox.layers.8.attention.masked_bias', 'gpt_neox.layers.14.attention.bias', 'gpt_neox.layers.4.attention.masked_bias', 'gpt_neox.layers.1.post_attention_layernorm.bias']