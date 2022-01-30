import argparse
import json
import numpy as np
import os

def combine_codes(args):
    result_files = os.listdir(args.root)
    tmp_codes = {}
   
    # load the results and combine them
    for r_file in result_files:
        path = os.path.join(args.root, r_file)
        if args.debug:
            print(path)
        elif "bleu" in path:
            continue
        elif "results.json" in path:
           continue
        elif "codes" in path and args.save not in path:
            with open(path, "r") as f:
                results = json.load(f)
            for res in results:
                tmp_codes[res] = results[res]
            continue
    with open(os.path.join(args.root, args.save), 'w') as f:
        json.dump(tmp_codes, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="print debugging statements",
                        action="store_true")
    parser.add_argument("--root", default="./results", type=str, help="which folder to merge the results")
    parser.add_argument("-s","--save", default="all_codes.json", type=str, help="Large final save file name. Note other files use the default value.")
    args = parser.parse_args()

    combine_codes(args)

if __name__ == "__main__":
    main()
