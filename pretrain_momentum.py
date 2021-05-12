from megatron.neox_arguments import NeoXArgs
from megatron.training import pretrain, get_momentum_model

if __name__ == "__main__":
    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer() # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    pretrain(neox_args=neox_args, model_provider=get_momentum_model)
