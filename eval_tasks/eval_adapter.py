# Copyright (c) 2024, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from megatron.utils import is_local_main, print_rank_0

import copy
import os
import sys
import itertools
import dataclasses
from functools import partial

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from tqdm import tqdm
import torch
import torch.nn.functional as F

from lm_eval.models.utils import chunks
from lm_eval.models.huggingface import HFLM
from lm_eval.loggers.utils import get_git_commit_hash
from lm_eval import tasks, evaluator, utils, api
from megatron.text_generation_utils import generate_samples_from_prompt
from megatron import mpu


class EvalHarnessAdapter(HFLM):
    """
    An adapter to run NeoX models on LM Evaluation Harness (https://github.com/EleutherAI/lm-evaluation-harness) tasks.

    Args:
        model: A NeoX Model
        forward_step_fn: A function that runs a forward pass through the model, returning `tuple(loss, logits)`.
        neox_args: a NeoXArgs object containing the model configuration.
        batch_size (optional): An argument to override the batch size, which defaults to batch size per gpu * dp world size.
    """

    def __init__(self, model, forward_step_fn, neox_args, batch_size=None):
        self.cache_hook = api.model.CacheHook(None)
        self._model = model
        self.neox_args = neox_args
        self.tokenizer = neox_args.tokenizer
        self._device = torch.device(f"cuda:{neox_args.local_rank}")
        self._eot_token_id = neox_args.tokenizer.eod_id
        self._max_length = neox_args.max_position_embeddings
        self._max_gen_toks = 128
        self._vocab_size = neox_args.padded_vocab_size

        # parallelism args:
        self.is_main = neox_args.rank == 0
        self.is_local_main = neox_args.local_rank == 0
        self.is_model_parallel = neox_args.model_parallel_size > 1
        self.is_pipe_parallel = self.model.is_pipe_parallel
        self.is_data_parallel = self.model.is_data_parallel
        self.is_last_stage = (
            True if not self.is_pipe_parallel else model.is_last_stage()
        )  # only the last stage of the pipeline model will receive the logits
        self.dp_world_size = mpu.get_data_parallel_world_size()
        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_group = mpu.get_data_parallel_group()
        self.is_mp_rank_0 = mpu.get_model_parallel_rank() == 0

        self._batch_size = batch_size or (
            neox_args.batch_size * self.dp_world_size
        )  # default batch size to bs per gpu * dp size
        # some utility functions:
        # we need to patch tokenizer methods, because lm_eval uses them internally:
        self.tokenizer.encode = self.tokenizer.tokenize
        self.tokenizer.decode = self.tokenizer.detokenize
        self._forward_step_fn = partial(
            forward_step_fn, neox_args=neox_args, timers=None, return_logits=True
        )
        self.generate = partial(
            generate_samples_from_prompt,
            neox_args=neox_args,
            model=model,
        )

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self._eot_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)

    def generate_until(self, requests):
        """
        Generate until is lm_eval harness' way to say "do greedy generation" - necessary for some tasks.
        the eval harness dispatches requests to the model, and the model does argmax generation, the results of which
        are returned to the eval harness to evaluate.

        TODO: batched / data parallel generation

        :param requests: Dictionary of requests containing the context (prompt) and 'until' - a token or
                         list of stop tokens.
        """
        self.model.module.inference_mode(use_cache=True)  # tell model to cache kv pairs
        res = []

        # get only the args from each Instance object
        reqs = [req.args for req in requests]

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (len(toks), x[0])

        reord = utils.Reorderer(reqs, _collate)
        for context, gen_kwargs in tqdm(
            reord.get_reordered(), "Running greedy generation"
        ):
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [kwargs]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {kwargs}"
                )
            if not until:
                until = [self.tok_decode(self.eot_token_id)]
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            if "do_sample" in kwargs.keys():
                kwargs.pop("do_sample")

            stop_tokens = [self.tokenizer.encode(i) for i in until]
            cont = self.generate(
                text=context,
                stop_tokens=stop_tokens,
                recompute=self.neox_args.recompute,
                maximum_tokens=max_gen_toks,
                **kwargs,
            )
            if cont:
                s = cont[0]["text"] or ""
            else:
                s = ""

            for term in until:
                s = s.split(term)[0]

            # partial caching
            self.cache_hook.add_partial("generate_until", (context, until), s)

            res.append(s)

        self.model.module.train_mode()  # set back to train mode
        return reord.get_original(res)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        """
        In this method, the model doesn't do any generation, but just returns log likelihoods
        for the next token, which eval harness uses to evaluate.

        :param requests: Dictionary of requests containing the context and the expected continuation.
        :param disable_tqdm: If True, disable tqdm progress bar.
        """
        self.model.module.inference_mode(
            use_cache=False
        )  # tell model to gather parallel outputs, but not cache key-value pairs

        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        with torch.no_grad():

            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in chunks(
                tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size
            ):
                inps, contlens, inplens, padding_length = [], [], [], None
                for cache_key, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                    ).to(self.device)
                    (inplen,) = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = (
                        padding_length if padding_length is not None else inplen
                    )

                    # pad to length
                    inp = torch.cat(
                        [
                            inp,  # [seq]
                            torch.zeros(padding_length - inplen, dtype=torch.long).to(
                                inp.device
                            ),  # [padding_length - seq]
                        ],
                        dim=0,
                    )

                    inps.append(inp.unsqueeze(0))
                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)

                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]
                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                        chunk, multi_logits, inps, inplens, contlens
                    ):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen : inplen].unsqueeze(
                            0
                        )  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = (
                            torch.tensor(cont_toks, dtype=torch.long)
                            .unsqueeze(0)
                            .to(multi_logits.device)
                        )
                        max_equal = (greedy_tokens == cont_toks).all()
                        logits = torch.gather(
                            logits, 2, cont_toks.unsqueeze(-1)
                        ).squeeze(
                            -1
                        )  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))

                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial(
                                "loglikelihood", cache_key, answer
                            )

                        res.append(answer)

            # broadcast results to all ranks
            if self.is_pipe_parallel:
                src_rank = self.model.grid.stage_to_global(self.model.num_stages - 1)
                if res:
                    logits_sums, max_equals = list(zip(*res))
                    logits_sums = torch.FloatTensor(logits_sums).cuda()
                    max_equals = torch.LongTensor(max_equals).cuda()
                else:
                    logits_sums = torch.zeros(res_len, dtype=torch.float32).cuda()
                    max_equals = torch.zeros(res_len, dtype=torch.int64).cuda()
                torch.distributed.broadcast(
                    tensor=logits_sums,
                    src=src_rank,
                    group=mpu.get_pipe_parallel_group(),
                )
                torch.distributed.broadcast(
                    tensor=max_equals, src=src_rank, group=mpu.get_pipe_parallel_group()
                )
                max_equals = [bool(i) for i in max_equals.tolist()]
                logits_sums = logits_sums.tolist()
                res = list(zip(logits_sums, max_equals))

        self.model.module.train_mode()  # set back to train mode
        return reord.get_original(res)

    def _dp_scatter(self, inps):
        """
        Scatters the inputs to all data parallel ranks.
        """

        batch_size = inps.shape[0]
        padded = False
        if batch_size % self.dp_world_size != 0:
            # The last batch could potentially not fill the full batch size (if the dataset size is not divisible by batch size)
            # In this case we pad the batch
            padded_size = self.dp_world_size - (batch_size % self.dp_world_size)

            print_rank_0(
                f"WARNING: Batch size ({batch_size}) must be divisible by dp world size ({self.dp_world_size}). Padding inputs to {padded_size}."
            )

            inps = torch.cat(
                [inps] + [inps[0:1, ...] for _ in range(padded_size)], dim=0
            )  # pad with first inp item
            padded = True

        assert (
            inps.shape[0] % self.dp_world_size == 0
        ), f"batch size ({inps.shape[0]}) must be divisible by dp world size ({self.dp_world_size})"

        # get a chunk for each data parallel rank
        chunk_size = inps.shape[0] // self.dp_world_size
        inps = inps[self.dp_rank * chunk_size : (self.dp_rank + 1) * chunk_size]
        # make a dummy dataloader / iterator to pass to model
        # we need to do this because deepspeed pipe parallel only takes an iterator
        # in this format
        return iter([{"text": F.pad(inps, pad=(0, 1))}]), padded

    def _dp_gather(self, logits):
        """
        Gather logits from all data parallel ranks
        """
        if logits is not None:
            tensor_list = [torch.zeros_like(logits) for _ in range(self.dp_world_size)]
            torch.distributed.all_gather(
                tensor_list, logits, group=mpu.get_data_parallel_group()
            )
            logits = torch.cat(tensor_list, dim=0)
            return logits

    def _model_call(self, inps):
        batch_size = inps.shape[0]

        # scatter inputs to all dp ranks:
        inps, padded = self._dp_scatter(inps)

        if self.neox_args.is_pipe_parallel:
            # need these flags to stop deepspeed pipe parallel from hanging
            self.model.first_output_send = True
            self.model.pipe_recv_buf = None

        _, logits = self._forward_step_fn(model=self.model, data_iterator=inps)

        # gather outputs from all dp ranks:
        logits = self._dp_gather(logits)

        # if logits have been padded (normally just last item where batch size is unequal)
        # restore to original shape
        if padded and logits is not None:
            logits = logits[:batch_size, ...]
        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override `greedy_until``.
        raise NotImplementedError()

    @torch.no_grad()
    def run_eval(
        self,
        eval_tasks=None,
        num_fewshot=0,
        bootstrap_iters=2,
        use_cache=True,
        name="neox",
        limit=None,
    ):
        was_training = self.model.training
        self.model.eval()
        in_micro_batches = (
            self.model.micro_batches
        )  # store input microbatches - we need to set to 1 during eval, but want to return to its original value after
        self.model.micro_batches = 1
        if eval_tasks is None:
            eval_tasks = [
                "lambada",
                "piqa",
                "hellaswag",
                "winogrande",
                "mathqa",
                "pubmedqa",
                "triviaqa",
            ]

        # register all the default tasks bundled with lm-evaluation-harness repository
        task_manager = tasks.TaskManager()
        task_manager.initialize_tasks()

        # Returns a list containing all values of the task registry that
        # match at least one of the patterns
        import fnmatch

        def pattern_match(patterns, source_list):
            task_names = set()
            for pattern in patterns:
                for matching in fnmatch.filter(source_list, pattern):
                    task_names.add(matching)
            return list(task_names)

        all_tasks = task_manager._all_tasks
        eval_tasks = pattern_match(eval_tasks, all_tasks)
        print(f"Found tasks: {eval_tasks}")

        assert len(eval_tasks) > 0, "Must run at least one task"

        # **HACK INCOMING**:
        # first get task dict on local main rank
        # the tasks are downloaded *as they are initialized*, and the downloads don't like multithreading.
        # so we download them once on the local main rank, wait, and then initialize them on all other ranks, which *should* load from the cache.
        if self.is_local_main:
            task_dict = tasks.get_task_dict(eval_tasks)
        # torch barrier
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        task_dict = tasks.get_task_dict(eval_tasks)

        lm = self

        if use_cache:
            use_cache = (
                "lm_cache/neox"
                + "_dp_rank"
                + str(self._dp_rank)
                + "_dp_group"
                + str(self._dp_group)
                + ".db"
            )
            print(f"Using cache at {use_cache}...")
            lm = lm_eval.api.model.CachingLM(
                lm,
                use_cache
                # each rank receives a different cache db.
                # necessary to avoid multiple writes to cache at once
                # TODO: Append a subset of `neox_args` to the cache database
                # name arg to distinguish model runs that use different configurations.
            )

        # from simple_evaluate:
        # override fewshot values for all tasks we can
        for task_name in task_dict.keys():
            group_task_objects = []
            top_level_task = task_dict[task_name]
            if isinstance(task_name, str):
                group_task_objects.append(top_level_task)
            else:
                for task_group in list(task_dict[task_name].values()):
                    group_task_objects.extend(list(task_group.values()))

            for task_obj in group_task_objects:
                if type(task_obj) == tuple:
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue

                config = task_obj._config

                utils.setup_logging()
                configs = []
                if num_fewshot is not None:
                    if config["num_fewshot"] == 0:
                        utils.logging.info(
                            f"num_fewshot has been set to 0 for {config.task} in its config. Manual configuration will be ignored."
                        )
                    else:
                        default_num_fewshot = config["num_fewshot"]
                        if not default_num_fewshot:
                            utils.logging.warning(
                                f"Overwriting default num_fewshot of {config.task} from {default_num_fewshot} to {num_fewshot}"
                            )

                        task_obj._config["num_fewshot"] = num_fewshot

        results = evaluator.evaluate(
            lm=lm,
            task_dict=task_dict,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
            log_samples=False,
        )

        results["config"] = {
            "model": name,
            "model_args": dataclasses.asdict(self.neox_args),
            "batch_size": self.batch_size,
            "device": str(self.device),
            "use_cache": use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
        }
        results["git_hash"] = get_git_commit_hash()

        print(results.keys())
        for task_name in task_dict.keys():
            sub_task_names = []
            if isinstance(task_name, str):
                sub_task_names.append(task_name)
            else:
                task_groups = task_dict[task_name]
                tasks_by_group = [list(group.keys()) for group in task_groups.values()]
                sub_task_names = list(itertools.chain(*tasks_by_group))

            for sub_task in sub_task_names:
                if "alias" in results["results"][sub_task]:
                    results["results"][sub_task].pop("alias")

        if was_training:
            self.model.train()
        self.model.micro_batches = in_micro_batches
        return results


def run_eval_harness(
    model,
    forward_step_fn,
    neox_args,
    batch_size=None,
    eval_tasks=None,
    num_fewshot=0,
    bootstrap_iters=2,
):
    print_rank_0("Running evaluation harness...")
    adapter = EvalHarnessAdapter(
        model, forward_step_fn, neox_args, batch_size=batch_size
    )
    return adapter.run_eval(
        eval_tasks=eval_tasks,
        num_fewshot=num_fewshot,
        bootstrap_iters=bootstrap_iters,
        use_cache=False,
        limit=neox_args.eval_task_limit
    )
