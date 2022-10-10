import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')

    args = parser.parse_args()
    csv = pd.read_csv(args.file)
    csv.drop_duplicates(subset='index', inplace=True)
    csv.to_csv(args.file, index=False)