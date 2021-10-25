import argparse

from eda import EDA


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--input", default="data", help="input file of unaugmented data")
    parser.add_argument("--output", default="output", help="output file of unaugmented data")
    parser.add_argument("--num_aug", type=int, default=9, help="number of augmented sentences per original sentence")
    parser.add_argument("--alpha_sr", type=float, default=0.1, help="percent of words in each sentence to be replaced by synonyms")
    parser.add_argument("--alpha_ri", type=float, default=0.1, help="percent of words in each sentence to be inserted")
    parser.add_argument("--alpha_rs", type=float, default=0.1, help="percent of words in each sentence to be swapped")
    parser.add_argument("--alpha_rd", type=float, default=0.1, help="percent of words in each sentence to be deleted")
    args = parser.parse_args()
    print(args)

    eda = EDA()
    eda.execute()
