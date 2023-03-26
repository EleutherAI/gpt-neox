# How to use Mup (https://github.com/microsoft/mup)

## Add mup neox args to your config

```
# mup

"use-mup": true,

"save-base-shapes": false, # this only needs to be enabled once in order to generate the base-shapes-file on each rank

"base-shapes-file": "base-shapes", # load base shapes from this file

"coord-check": false, # generate coord check plots to verify mup's implementation in neox

# mup hp search

"mup-init-scale": 1.0,

"mup-attn-temp": 1.0,

"mup-output-temp": 1.0,

"mup-embedding-mult": 1.0,

"mup-rp-embedding-mult": 1.0,
```

## Generate base shapes

1. Set use-mup to true
2. Set save-base-shapes to true
3. Run once. gpt-neox will instantiate a base model and a delta model, then save one file per rank named <base-shapes-file>.<rank>. gpt-neox will exit immediately.
4. Set save-base-shapes to false

## Generate coord check plots (optional)

1. Keep use-mup true
2. Set coord-check to true
3. Run once. gpt-neox will output jpg images similar to https://github.com/microsoft/mutransformers/blob/main/README.md#coord-check. gpt-neox will exit immediately
4. Set coord-check to false

## Tune mup hyperparameters and LR

The values under `mup hp search` were added and correspond to appendix F.4 from https://arxiv.org/pdf/2203.03466.pdf. These and LR are tuned with a random search using the scaled-up config (tested with 6-7B.yml) but with hidden-size set to the value from the scaled-down config (125M.yml).

## Transfer

With the best LR set and the best mup HPs set, revert the value of hidden-size in the scaled-up config and run again.
