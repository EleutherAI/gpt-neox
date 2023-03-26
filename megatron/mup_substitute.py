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

from mup import coord_check as mup_coord_check
from megatron.training import train_step


def _get_coord_data(
    neox_args,
    timers,
    lr_scheduler,
    models,
    dataloader,
    optcls,
    nsteps=3,
    dict_in_out=False,
    flatten_input=False,
    flatten_output=False,
    output_name="loss",
    lossfn="xent",
    filter_module_by_name=None,
    fix_data=True,
    cuda=True,
    nseeds=1,
    output_fdict=None,
    input_fdict=None,
    param_fdict=None,
    show_progress=True,
    one_hot_target=False,
):
    df = []

    for i in range(nseeds):
        torch.manual_seed(i)
        for width, model in models.items():
            model = model()
            model.train()
            optimizer = optcls(model)
            for step in range(nsteps + 1):
                remove_hooks = []
                # add hooks
                for name, module in model.named_modules():
                    if filter_module_by_name and not filter_module_by_name(name):
                        continue
                    remove_hooks.append(
                        module.register_forward_hook(
                            mup_coord_check._record_coords(
                                df,
                                width,
                                name,
                                step + 1,
                                output_fdict=output_fdict,
                                input_fdict=input_fdict,
                                param_fdict=param_fdict,
                            )
                        )
                    )

                # train for a step
                loss_dict, skipped_iter = train_step(
                    neox_args=neox_args,
                    timers=timers,
                    data_iterator=dataloader,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )

                # remove hooks
                for handle in remove_hooks:
                    handle.remove()

            import gc

            del model
            gc.collect()

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
    if mup:
        from mup.optim import MuAdam as Adam
        from mup.optim import MuAdamW as AdamW
        from mup.optim import MuSGD as SGD
    else:
        from torch.optim import SGD, Adam, AdamW

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
    data["optimizer"] = optimizer
    data["lr"] = lr
    return data
