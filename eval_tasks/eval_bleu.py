"""
Eval outpus via BLEU measure.
"""

import json
import logging
import math
import numpy as np
import os
import pprint
import random
import sys
import time

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from typing import List

#bleu imports
import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang='en')

random.seed(12345678987654321)

def calc_bleu(output:List[str], targets:List[List[str]]):
    max_bleu = 0
    bleu = sacrebleu.corpus_bleu(output, targets)
    for item in targets[0]:
        tmp_bleu = sacrebleu.corpus_bleu(output, [[item]])
        if tmp_bleu.score > max_bleu:
            max_bleu = tmp_bleu.score
    return bleu.score, max_bleu

def eval_and_save_bleu_scores(args):
    with open(args.test_loc, "r") as f:
        problems = json.load(f)

    gpt_codes = {}
    gpt_bleu = {}
    codes_loc = os.path.join(args.save, f"all_codes.json")
    if not os.path.exists(codes_loc):
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")
 
    if os.path.exists(codes_loc):
        with open(codes_loc, "r") as f:
            gpt_codes = json.load(f)

    if args.index:
        problems = [problems[args.index]]
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = problems[start:end]

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        prob_path = os.path.join(args.root, problem)
        if args.debug:
            print(f"problem path = {problem}")
        try:
            output_strs = gpt_codes[str(index+args.start)]
        except:
            continue

        if args.debug:
            print(output_str)

        with open(os.path.join(prob_path, "solutions.json"), "r") as f:
            sols = json.load(f)

        random.shuffle(sols)
        if args.debug:
            sols = sols[:100]

        tmp = []
        for sol in sols:
            tmp.append([sol])
        
        sols = tmp

        # this is if we generated multiple outputs per problem
        if isinstance(output_strs, list):
            gpt_bleu[index+args.start] = []
            for output_str in output_strs:
                gpt_bleu[index+args.start].extend(calc_bleu([output_str], sols))
        # one output per problem
        else:
            output_str = output_strs
            gpt_bleu[index+args.start] = calc_bleu([output_str], sols)

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        if args.end is None and args.index is None:
            bleu_loc = os.path.join(args.save, f"all_bleu_results.json")
        elif args.index:
            bleu_loc = os.path.join(args.save, f"{args.index}_bleu_results.json")
        else:
            bleu_loc = os.path.join(args.save, f"{args.start}-{args.end}_bleu_results.json") 

        with open(bleu_loc, "w") as f:
            json.dump(gpt_bleu, f)

    return gpt_bleu

def print_results(results):
    bleu_scores = []
    max_bleu_scores = []
    for res in results:
        bleu_scores.append(results[res][0])
        max_bleu_scores.append(results[res][1])

    print(f"Average BLEU Score = {np.mean(bleu_scores)}")
    print(f"Average of Max BLEU Score = {np.mean(max_bleu_scores)}")


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if args.print_results:
        bleu_loc = os.path.join(args.save, f"all_bleu_results.json")
        if os.path.exists(bleu_loc):
            with open(bleu_loc, "r") as f:
                results = json.load(f)
        else:
            print(f"Error file does not exist in this path {bleu_loc}. Exiting.")
            return
    else:
        results = eval_and_save_bleu_scores(args)

    print_results(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BLEU Evaluation")
    parser.add_argument("-t","--test_loc", default="../data_split/test.json", type=str)
    parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-p", "--print_results", action="store_true", help="If you have already evaluated the results and only want to print them.")
    parser.add_argument("--save", type=str, default="./results")
 
    args = parser.parse_args()

    main(args)
