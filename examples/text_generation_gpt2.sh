# Run from gpt-neox root directory

# Generate text from a pre-saved model either unconditionally or from input file.
# Switch between the two by setting `text-gen-type` in `configs/text_generation.yml`
./deepy.py pretrain_gpt2.py -d configs small.yml eleutherai_cluster.yml text_generation.yml
