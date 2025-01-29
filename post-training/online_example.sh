# Launch vllm
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n vllm python -m vllm.entrypoints.openai.api_server --model=meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --from-remote-program --tensor-parallel-size=4 --enforce-eager --gpu-memory-utilization=0.2 --port 8000 --max-model-len=1024 --max-num-seqs=512 &

CUDA_VISIBLE_DEVICES=4,5,6,7 conda run --no-capture-output -n vllm python -m vllm.entrypoints.openai.api_server --model=meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --from-remote-program --tensor-parallel-size=4 --enforce-eager --gpu-memory-utilization=0.2 --port 8001 --max-model-len=1024 --max-num-seqs=512 &

# Launch training
conda run --no-capture-output -n neox python deepy.py train.py post-training/configs/llama3-8b-reinforce.yml
