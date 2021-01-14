mkdir logs
NCCL_SHM_DISABLE=1 NCCL_DEBUG=info MASTER_ADDR=127.0.0.1 MASTER_PORT=2000 deepspeed train.py --deepspeed --deepspeed_config configs/deepspeed_zero1.json
