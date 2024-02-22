"""
Helper functions for performing coord check.
"""
import os
from copy import copy
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from megatron import print_rank_0
from megatron.training import train_step


def _get_coord_data(
    neox_args,
    timers,
    lr_scheduler,
    models,
    dataloader,
    optcls,
    nsteps=10,
    dict_in_out=False,
    flatten_input=False,
    flatten_output=False,
    output_name="loss",
    lossfn="xent",
    filter_module_by_name=None,
    fix_data=True,
    cuda=True,
    nseeds=2,
    output_fdict=None,
    input_fdict=None,
    param_fdict=None,
    show_progress=True,
    one_hot_target=False,
):
    df = {
        "seed": [],
        "step": [],
        "word_embedding_act_abs_mean": [],
        "attn_output_act_abs_mean": [],
        "ffn_output_act_abs_mean": [],
        "output_logits_act_abs_mean": [],
        "width": [],
    }

    for width, model_obj in models.items():
        for i in range(nseeds):
            torch.manual_seed(10**i)
            print_rank_0(f">>> Running Model with width: {width} on seed: {i}")
            model = model_obj()
            model.train()
            neox_args.hidden_size = width
            optimizer = optcls(model)

            for step in range(nsteps + 1):

                word_embedding_act_abs_mean_list = []
                attn_output_act_abs_mean_list = []
                ffn_output_act_abs_mean_list = []
                output_logits_act_abs_mean_list = []
                remove_hooks = []

                def word_embedding_coord_check_hook(module, input, output):
                    with torch.no_grad():
                        word_embedding_act_abs_mean_list.append(output.abs().mean().item())

                def attn_output_coord_check_hook(module, input, output):
                    with torch.no_grad():
                        attn_output_act_abs_mean_list.append(output[0].abs().mean().item())

                def ffn_output_coord_check_hook(module, input, output):
                    with torch.no_grad():
                        ffn_output_act_abs_mean_list.append(output[0].abs().mean().item())

                def output_logits_coord_check_hook(module, input, output):
                    with torch.no_grad():
                        # print("output_logits_coord_check_hook")
                        # print_rank_0(output.shape)
                        output_logits_act_abs_mean_list.append(output[0].abs().mean().item())

                for name, module in model.named_modules():
                    print_rank_0(name)
                    if name.endswith(".word_embeddings"):
                        remove_hooks.append(
                            module.register_forward_hook(word_embedding_coord_check_hook)
                        )
                    elif name.endswith(".attention.dense"):
                        remove_hooks.append(
                            module.register_forward_hook(attn_output_coord_check_hook)
                        )
                    elif name.endswith(".mlp.dense_4h_to_h"):
                        remove_hooks.append(
                            module.register_forward_hook(ffn_output_coord_check_hook)
                        )
                    elif name.endswith(".final_linear"):
                        remove_hooks.append(
                            module.register_forward_hook(output_logits_coord_check_hook)
                        )

                # train for a step
                train_step(
                    neox_args=neox_args,
                    timers=timers,
                    data_iterator=dataloader,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )

                word_embedding_act_abs_mean = None
                attn_output_act_abs_mean = None
                ffn_output_act_abs_mean = None
                output_logits_act_abs_mean = None

                # remove hooks
                for handle in remove_hooks:
                    handle.remove()
                word_embedding_act_abs_mean = np.mean(word_embedding_act_abs_mean_list)
                attn_output_act_abs_mean = np.mean(attn_output_act_abs_mean_list)
                ffn_output_act_abs_mean = np.mean(ffn_output_act_abs_mean_list)
                output_logits_act_abs_mean = np.mean(output_logits_act_abs_mean_list)

                df["seed"].append(i)
                df["step"].append(f"t={step}")
                df["word_embedding_act_abs_mean"].append(word_embedding_act_abs_mean)
                df["attn_output_act_abs_mean"].append(attn_output_act_abs_mean)
                df["ffn_output_act_abs_mean"].append(ffn_output_act_abs_mean)
                df["output_logits_act_abs_mean"].append(output_logits_act_abs_mean)
                df["width"].append(width)

            import gc
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()

    return pd.DataFrame(df)


def get_coord_data(
    neox_args,
    timers,
    lr_scheduler,
    models,
    dataloader,
    optimizer="sgd",
    lr=None,
    mup=True,
    filter_trainable_by_name=None,
    **kwargs
):
    """Get coord data for coord check.
    Train the models in `models` with data from `dataloader` and optimizer
    specified by `optimizer` and `lr` for `nsteps` steps, and record coordinate
    statistics specified by `output_fdict`, `input_fdict`, `param_fdict`. By
    default, only `l1` is computed for output activations of each module.
    This function wraps around `_get_coord_data`, with the main difference being
    user can specify common optimizers via a more convenient interface.
    Inputs:
        models:
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optimizer:
            a string in `['sgd', 'adam', 'adamw']`, with default being `'sgd'`.
        lr:
            learning rate. By default is 0.1 for `'sgd'` and 1e-3 for others.
        mup:
            If True, then use the optimizer from `mup.optim`; otherwise, use the
            one from `torch.optim`.
        filter_trainable_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be trained.
        nsteps:
            number of steps to train the model
        dict_in_out:
            whether the data loader contains Huggingface-style dict input and
            output. Default: False
        flatten_input:
            if not `dict_in_out`, reshape the input to be
            `input.view(input.shape[0], -1)`. Typically used for testing MLPs.
        flatten_output:
            if not `dict_in_out`, reshape the label to be `label.view(-1,
            input.shape[-1])`.
        output_name:
            if `dict_in_out`, this is the key for the loss value if the output
            is a dict. If the output is not a dict, then we assume the first
            element of the output is the loss.
        lossfn:
            loss function to use if not `dict_in_out`. Can be either a string from
            [`xent`, 'mse', 'nll', 'l1'] or a python `callable` such that
            `lossfn(output, target)` returns the loss value. Examples of valid
            `callable`s are `F.cross_entropy`, `F.mse_loss`, etc, where `F` is
            `torch.nn.functional`. Default: 'xent'
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict:
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm. Default: True
        one_hot_target:
            convert target label into a one-hot vector. This typically is only
            used for `'mse'` or `'l1'` losses in classification tasks.
            Default: False
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).

    Breaking Changes:
        In v1.0.0, when `lossfn=='mse'`, the target is automatically converted
        to a one hot vector before loss computation. Starting in v1.1.0, this
        behavior is turned off, and the user needs to explicitly turn on this
        behavior by setting `one_hot_target=True`.
    """
    if lr is None:
        lr = 0.1 if optimizer == "sgd" else 1e-3
        
    from torch.optim import SGD, AdamW, Adam
    # from deepspeed.ops.adam import FusedAdam as Adam

    def get_trainable(model):
        params = model.parameters()
        if filter_trainable_by_name is not None:
            params = []
            for name, p in model.named_parameters():
                if filter_trainable_by_name(name):
                    params.append(p)
        return params

    if optimizer == "sgd":
        optcls = lambda model: SGD(get_trainable(model), lr=lr)
    elif optimizer == "adam":
        optcls = lambda model: Adam(get_trainable(model), lr=lr)
    elif optimizer == "adamw":
        optcls = lambda model: AdamW(get_trainable(model), lr=lr)
    elif optimizer is None:
        raise ValueError("optimizer should be sgd|adam|adamw or a custom function")

    data = _get_coord_data(
        neox_args, timers, lr_scheduler, models, dataloader, optcls, **kwargs
    )
    # data["optimizer"] = optimizer
    # data["lr"] = lr
    return data
