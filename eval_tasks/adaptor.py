import os
import sys
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import print_rank_0
from megatron.training import forward_step
from megatron.initialize import initialize_megatron
from megatron.neox_arguments import NeoXArgs
from megatron import initialize_megatron
from megatron.training import setup_model_and_optimizer
from megatron.utils import print_rank_0, is_mp_rank_0, is_local_main
from megatron.utils import ddb

from tqdm import tqdm 

import torch
import lm_eval
from lm_eval.base import CacheHook

from lm_eval.models.gpt2 import GPT2LM
from lm_eval import tasks, evaluator
from lm_eval import utils
import torch.nn.functional as F

import collections
import itertools
import random
import lm_eval.metrics

class EvalHarnessAdaptor(GPT2LM):

    def __init__(self, model, forward_step_fn, neox_args, batch_size=None):
        self.device = torch.device(f'cuda:{neox_args.local_rank}')
        self.VOCAB_SIZE = neox_args.padded_vocab_size
        self.tokenizer = neox_args.tokenizer
        self.EOT_TOKEN_ID = neox_args.tokenizer.eod_id
        self.model = model
        self._forward_step_fn = partial(forward_step_fn, neox_args=neox_args, timers=None, return_logits=True)
        self.max_length = neox_args.max_position_embeddings // 2
        self.tokenizer.encode = self.tokenizer.tokenize  # patch tokenizer encode method
        self.batch_size = batch_size or neox_args.batch_size
        self.neox_args = neox_args
        self.cache_hook = CacheHook(None)
        self.is_main = neox_args.rank == 0
        self.is_local_main = neox_args.local_rank == 0
        self.is_pipe_parallel = neox_args.pipe_parallel_size > 1
        self.is_last_stage = True if not self.is_pipe_parallel else model.is_last_stage() # only the last stage of the pipeline model will receive the logits

    def greedy_until(self, requests):
        raise NotImplementedError

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0 # storing the result length for later
        with torch.no_grad():

            def _collate(x):
                # the negative sign on len(toks) sorts descending - this has a few advantages:
                # - time estimates will always be over not underestimates, which is more useful for planning
                # - to know the size of a batch when going through the list, you know the first one is always the batch padded context length.
                #   this is useful to simplify the batching logic and more importantly to make automatic adaptive batches much much easier to implement
                # - any OOMs will happen right away rather than near the end

                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))
            
            # TODO: automatic (variable) batch size detection for vectorization
            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps = []
                contlens = []
                inplens = []

                padding_length = None

                # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
                # tensors, then we pack them together into a batch, call the model, and then pick it all apart
                # again because vectorizing is annoying

                for _, context_enc, continuation_enc in chunk:
                    # sanity check
                    assert len(context_enc) > 0
                    assert len(continuation_enc) > 0
                    assert len(continuation_enc) <= self.max_length

                    # how this all works:
                    #          CTX      CONT
                    # inp    0 1 2 3|4 5 6 7 8 9 <- last token is deleted by inp[:, :-1]
                    # gpt2    \               \
                    # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the [:, -len(continuation_enc):, :self.VOCAB_SIZE] slice
                    # cont_toks      4 5 6 7 8 9

                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length+1):][:-1]
                    , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen

                    # pad to length
                    inp = torch.cat([
                        inp, # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device) # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))
                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)

                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]
                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, contlens):
                        contlen = len(cont_toks)

                        logits = logits[inplen-contlen:inplen].unsqueeze(0) # [1, seq, vocab]

                        greedy_tokens = logits.argmax(dim=-1)

                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0).to(multi_logits.device)

                        max_equal = (greedy_tokens == cont_toks).all()

                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1) # [1, seq]
                        
                        logits_sum = logits.sum()

                        answer = (float(logits.sum()), bool(max_equal))

                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                        res.append(answer)

        # broadcast results
        if self.is_pipe_parallel:
            src_rank = self.model.grid.stage_to_global(self.model.num_stages - 1)
            if res:
                logits_sums, max_equals = list(zip(*res))
                logits_sums = torch.FloatTensor(logits_sums).cuda()
                max_equals = torch.LongTensor(max_equals).cuda()
            else:
                logits_sums = torch.zeros(res_len, dtype=torch.float32).cuda()
                max_equals = torch.zeros(res_len, dtype=torch.int64).cuda()
            torch.distributed.broadcast(tensor=logits_sums, src=src_rank)
            torch.distributed.broadcast(tensor=max_equals, src=src_rank)
            max_equals = [bool(i) for i in max_equals.tolist()]
            logits_sums = logits_sums.tolist()
            res = list(zip(logits_sums, max_equals))

        return reord.get_original(res)

    def _model_call(self, inps):
        data_wrapped = iter([{'text': F.pad(inps, pad=(0, 1))}])
        if self.neox_args.is_pipe_parallel:
            # need these flags to stop deepspeed from hanging
            self.model.first_output_send = True
            self.model.pipe_recv_buf = None
        _, logits = self._forward_step_fn(model=self.model, data_iterator=data_wrapped)
        return logits 

    def run_eval(self, eval_tasks=None):
        was_training = self.model.training
        self.model.eval()
        if eval_tasks is None:
            eval_tasks = ["lambada", "piqa", "hellaswag", "winogrande", "mathqa", "pubmedqa"]
        results = evaluator.evaluate(lm=self, 
                                     task_dict=tasks.get_task_dict(eval_tasks), 
                                     provide_description=False, 
                                     num_fewshot=0, 
                                     limit=None,
                                     bootstrap_iters=1)
        if was_training:
            self.model.train()

def run_eval_harness(model, forward_step_fn, neox_args, batch_size=None):
    print_rank_0('Running evaluation harness...')
    adaptor = EvalHarnessAdaptor(model, forward_step_fn, neox_args, batch_size)
    return adaptor.run_eval()
