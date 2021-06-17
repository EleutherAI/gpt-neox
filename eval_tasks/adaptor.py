from megatron.utils import is_local_main 
import best_download

# patch best_download (eval harness downloader) to only happen on the first rank
fn = best_download.download_file

def _download_file(*args, **kwargs):
    if is_local_main():
        fn(*args, **kwargs)

best_download.download_file = _download_file

import os
import sys
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from megatron.utils import print_rank_0
from tqdm import tqdm
import torch
from lm_eval.base import CacheHook
from lm_eval.models.gpt2 import GPT2LM
from lm_eval import tasks, evaluator, utils
import torch.nn.functional as F
from megatron.text_generation_utils import generate_samples_from_prompt
import inspect 
from lm_eval import tasks 
from lm_eval.utils import chunks

# TODO: add data parallel

class EvalHarnessAdaptor(GPT2LM):

    def __init__(self, model, forward_step_fn, neox_args, batch_size=None):
        self.device = torch.device(f'cuda:{neox_args.local_rank}')
        self.VOCAB_SIZE = neox_args.padded_vocab_size
        self.tokenizer = neox_args.tokenizer
        self.EOT_TOKEN_ID = neox_args.tokenizer.eod_id
        self.model = model
        self._forward_step_fn = partial(forward_step_fn, neox_args=neox_args, timers=None, return_logits=True)
        self.max_length = neox_args.max_position_embeddings // 2
        self.max_gen_toks = 128
        self.tokenizer.encode = self.tokenizer.tokenize  # patch tokenizer encode + decode methods
        self.tokenizer.decode = self.tokenizer.detokenize
        self.batch_size = batch_size or neox_args.batch_size
        self.neox_args = neox_args
        self.cache_hook = CacheHook(None)
        self.is_main = neox_args.rank == 0
        self.is_local_main = neox_args.local_rank == 0
        self.is_pipe_parallel = self.model.is_pipe_parallel
        self.is_data_parallel = self.model.is_data_parallel
        self.is_last_stage = True if not self.is_pipe_parallel else model.is_last_stage()  # only the last stage of the pipeline model will receive the logits
        self.generate = partial(generate_samples_from_prompt, neox_args=neox_args, model=model, maximum_tokens=self.max_gen_toks)


    def greedy_until(self, requests, batch=False):
        self.model.module.inference_mode()
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (len(toks), x[0])

        reord = utils.Reorderer(requests, _collate)
        for context, until in tqdm(reord.get_reordered()):
            if isinstance(until, str): 
                until = [until]
            stop_tokens = [self.tokenizer.encode(i) for i in until]
            cont = self.generate(text=context, 
                                stop_tokens=stop_tokens, 
                                recompute = self.neox_args.recompute)

            s = cont[0]['text'] or ''

            for term in until:
                s = s.split(term)[0]
            
            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)

            res.append(s)

        self.model.module.train_mode()
        return reord.get_original(res)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        with torch.no_grad():

            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:

                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen

                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))
                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)

                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]
                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens,
                                                                                 contlens):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0).to(multi_logits.device)
                        max_equal = (greedy_tokens == cont_toks).all()
                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))

                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial("loglikelihood", cache_key, answer)

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
        in_micro_batches = self.model.micro_batches # store input microbatches - we need to set to 1 during eval
        self.model.micro_batches = 1
        if eval_tasks is None:
            eval_tasks = ["lambada", "piqa", "hellaswag", "winogrande", "mathqa", "pubmedqa"]
        results = evaluator.evaluate(lm=self,
                                        task_dict=tasks.get_task_dict(eval_tasks),
                                        provide_description=False,
                                        num_fewshot=0,
                                        limit=None,
                                        bootstrap_iters=2).get('results')
        if was_training:
            self.model.train()
        self.model.micro_batches = in_micro_batches
        return results


def run_eval_harness(model, forward_step_fn, neox_args, batch_size=None, eval_tasks=None):
    print_rank_0('Running evaluation harness...')
    adaptor = EvalHarnessAdaptor(model, forward_step_fn, neox_args, batch_size)
    return adaptor.run_eval(eval_tasks=eval_tasks)
