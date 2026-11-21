# Copyright (c) 2025, EleutherAI
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

import dataclasses
from functools import partial

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval import evaluator, utils
from lm_eval.api.model import TemplateLM
from lm_eval.loggers.utils import get_git_commit_hash
from lm_eval.tasks import TaskManager

from megatron import mpu
from megatron.text_generation_utils import generate_samples_from_prompt
from megatron.utils import print_rank_0


class EvalHarnessAdapter(TemplateLM):
    """Runs a NeoX model on lm-evaluation-harness tasks."""

    def __init__(self, model, forward_step_fn, neox_args, batch_size=None):
        super().__init__()
        self.model = model
        self.neox_args = neox_args
        self.tokenizer = neox_args.tokenizer
        self._device = torch.device(f"cuda:{neox_args.local_rank}")
        self.max_length = neox_args.max_position_embeddings
        self.max_gen_toks = 128

        self.is_main = neox_args.rank == 0
        self.is_local_main = neox_args.local_rank == 0
        self.is_pipe_parallel = getattr(model, "is_pipe_parallel", False)
        self.dp_world_size = mpu.get_data_parallel_world_size()
        self.dp_rank = mpu.get_data_parallel_rank()
        self.batch_size = batch_size or neox_args.batch_size * self.dp_world_size

        self._forward_step_fn = partial(
            forward_step_fn, neox_args=neox_args, timers=None, return_logits=True
        )
        self.generate = partial(
            generate_samples_from_prompt, neox_args=neox_args, model=model
        )

    @property
    def eot_token_id(self):
        return self.tokenizer.eod_id

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.tokenize(string)

    def generate_until(self, requests, disable_tqdm=False):
        self.model.module.inference_mode(use_cache=True)
        res = []
        for context, gen_kwargs in tqdm(
            [req.args for req in requests], disable=disable_tqdm or not self.is_main
        ):
            kwargs = dict(gen_kwargs)
            until = kwargs.pop("until", None) or []
            if isinstance(until, str):
                until = [until]
            max_gen_toks = kwargs.pop("max_gen_toks", self.max_gen_toks)
            kwargs.pop("do_sample", None)
            cont = self.generate(
                text=context,
                stop_tokens=[self.tok_encode(s) for s in until] or None,
                recompute=self.neox_args.recompute,
                maximum_tokens=max_gen_toks,
                **kwargs,
            )
            s = (cont[0]["text"] or "") if cont else ""
            for term in until:
                s = s.split(term)[0]
            res.append(s)
        self.model.module.train_mode()
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        windows, counts = [], []
        for (string,) in [req.args for req in requests]:
            doc_windows = [
                (None,) + utils.make_disjoint_window(w)
                for w in utils.get_rolling_token_windows(
                    token_list=self.tok_encode(string),
                    prefix_token=self.prefix_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                )
            ]
            windows += doc_windows
            counts.append(len(doc_windows))
        lls = iter(ll for ll, _ in self._loglikelihood_tokens(windows, disable_tqdm))
        return [sum(next(lls) for _ in range(n)) for n in counts]

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        self.model.module.inference_mode(use_cache=False)
        # longest first, so the first batch shows any OOM
        order = sorted(
            range(len(requests)), key=lambda i: -len(requests[i][1] + requests[i][2])
        )
        res = [None] * len(requests)
        for start in tqdm(
            range(0, len(order), self.batch_size),
            disable=disable_tqdm or not self.is_main,
        ):
            idx = order[start : start + self.batch_size]
            inps = [
                (requests[i][1] + requests[i][2])[-(self.max_length + 1) :][:-1]
                for i in idx
            ]
            padding_length = max(map(len, inps))
            batch = torch.tensor(
                [inp + [0] * (padding_length - len(inp)) for inp in inps],
                dtype=torch.long,
                device=self.device,
            )
            logits = self._model_call(batch)
            if logits is None:
                for i in idx:
                    res[i] = (0.0, False)
                continue
            logprobs = F.log_softmax(logits, dim=-1)
            for i, lp, inp in zip(idx, logprobs, inps):
                cont = torch.tensor(requests[i][2], dtype=torch.long, device=lp.device)
                lp = lp[len(inp) - len(cont) : len(inp)]
                res[i] = (
                    lp.gather(1, cont.unsqueeze(-1)).sum().item(),
                    bool((lp.argmax(dim=-1) == cont).all()),
                )

        if self.is_pipe_parallel:
            # only the last pipeline stage has logits
            src = self.model.grid.stage_to_global(self.model.num_stages - 1)
            group = mpu.get_pipe_parallel_group()
            lls = torch.tensor([r[0] for r in res], dtype=torch.float64).cuda()
            greedy = torch.tensor([r[1] for r in res], dtype=torch.long).cuda()
            torch.distributed.broadcast(lls, src=src, group=group)
            torch.distributed.broadcast(greedy, src=src, group=group)
            res = list(zip(lls.tolist(), map(bool, greedy.tolist())))

        self.model.module.train_mode()
        return res

    def _model_call(self, inps):
        # every rank gets every request, so split each batch over dp ranks
        batch_size = inps.shape[0]
        # pad with copies of the first row so the batch splits evenly over dp ranks
        padded_size = -batch_size % self.dp_world_size
        inps = torch.cat([inps] + [inps[:1]] * padded_size)
        chunk = inps.shape[0] // self.dp_world_size
        inps = inps[self.dp_rank * chunk : (self.dp_rank + 1) * chunk]
        data_iterator = iter([{"text": F.pad(inps, pad=(0, 1))}])

        if self.neox_args.is_pipe_parallel:
            # stops deepspeed pipe parallel from hanging
            self.model.first_output_send = True
            self.model.pipe_recv_buf = None
            _, logits = self._forward_step_fn(
                model=self.model, data_iterator=data_iterator
            )
        else:
            _, logits, _ = self._forward_step_fn(
                model=self.model, data_iterator=data_iterator
            )

        if logits is None:
            return None
        gathered = [torch.zeros_like(logits) for _ in range(self.dp_world_size)]
        torch.distributed.all_gather(
            gathered, logits, group=mpu.get_data_parallel_group()
        )
        return torch.cat(gathered)[:batch_size]

    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=0, bootstrap_iters=2, limit=None):
        was_training = self.model.training
        self.model.eval()
        in_micro_batches = getattr(self.model, "micro_batches", 1)
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

        task_manager = TaskManager()
        task_names = task_manager.match_tasks(eval_tasks)
        assert task_names, f"No lm_eval tasks match {eval_tasks}"
        print_rank_0(f"Found tasks: {task_names}")

        # datasets download when tasks load, so the local main rank loads first
        if not self.is_local_main:
            torch.distributed.barrier()
        task_dict = task_manager.load(task_names)
        if self.is_local_main:
            torch.distributed.barrier()

        for task in task_dict["tasks"].values():
            if num_fewshot is not None and task.get_config("num_fewshot") != 0:
                task.set_config(key="num_fewshot", value=num_fewshot)
            elif task.get_config("num_fewshot") is None:
                task.set_config(key="num_fewshot", value=0)

        results = evaluator.evaluate(
            lm=self,
            task_dict=task_dict,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
            log_samples=False,
        )
        # keep numeric values so they can be logged
        results["results"] = {
            task: {k: v for k, v in r.items() if isinstance(v, (int, float))}
            for task, r in results["results"].items()
        }
        results["config"] = {
            "model": "neox",
            "model_args": dataclasses.asdict(self.neox_args),
            "batch_size": self.batch_size,
            "device": str(self.device),
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
        }
        results["git_hash"] = get_git_commit_hash()

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
    limit=None,
):
    print_rank_0("Running evaluation harness...")
    adapter = EvalHarnessAdapter(
        model, forward_step_fn, neox_args, batch_size=batch_size
    )
    return adapter.run_eval(
        eval_tasks=eval_tasks,
        num_fewshot=num_fewshot,
        bootstrap_iters=bootstrap_iters,
        limit=limit,
    )
