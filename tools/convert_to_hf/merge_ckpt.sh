#SBATCH --partition=gpu 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=2 
#SBATCH --comment eleuther

cd /fsx/multi-lingual-6b/gpt-neox

model_size='3B' #1B, 3B, 20B, 39B

input_dir="./checkpoints/${model_size}_scratch"
output_dir="/fsx/multi-lingual-6b/merged_huggingface/${model_size}"
config_file="./configs/${model_size}_ko.yml"

tmp_output_dir="/fsx/multi-lingual-6b/merged_ckpt/${model_size}"

vocab_size=30080
vocab_file_path="./tokenizer/MBBPE/"

mkdir -p $output_dir
mkdir -p $tmp_output_dir

python3 tools/convert_to_hf/merge_neox.py --input_dir ${input_dir} --output_dir ${tmp_output_dir} --vocab_size ${vocab_size} --config_file ${config_file}
merged_steps=`cat ${tmp_output_dir}/latest`

python3 tools/convert_to_hf/merge_layers_and_convert_to_hf.py --output_dir ${output_dir} --input_dir ${tmp_output_dir}/${merged_steps}
python3 tools/convert_to_hf/make_hf_config.py --output_dir ${output_dir} --vocab_size ${vocab_size} --config_file ${config_file}

cp -r ${vocab_file_path}/* ${output_dir}