import argparse
import random
from enum import Enum

from eda_ja.eda import Eda, EdaEn, EdaJa


class Lang(Enum):
    EN = "en"
    JA = "ja"

    def __str__(self):
        return self.value


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--input", default="input.txt", help="input file of unaugmented data"
    )
    parser.add_argument(
        "--output", default="output.txt", help="output file of augmented data"
    )
    parser.add_argument(
        "--num_aug",
        type=int,
        default=9,
        help="number of augmented sentences per original sentence",
    )
    parser.add_argument(
        "--alpha_sr",
        type=float,
        default=0.1,
        help="percent of words in each sentence to be replaced by synonyms",
    )
    parser.add_argument(
        "--alpha_ri",
        type=float,
        default=0.1,
        help="percent of words in each sentence to be inserted",
    )
    parser.add_argument(
        "--alpha_rs",
        type=float,
        default=0.1,
        help="percent of words in each sentence to be swapped",
    )
    parser.add_argument(
        "--alpha_rd",
        type=float,
        default=0.1,
        help="percent of words in each sentence to be deleted",
    )
    parser.add_argument(
        "--stop_words_path",
        default="default",
        help="file path to stop words",
    )
    parser.add_argument(
        "--wordnet_path", default="wnjpn.db", help="file path to wordnet db"
    )
    parser.add_argument(
        "--lang", type=Lang, default=Lang.JA, choices=list(Lang), help="lang"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)

    instance: Eda
    if args.lang == Lang.EN:
        instance = EdaEn(args.stop_words_path)
    elif args.lang == Lang.JA:
        instance = EdaJa(args.stop_words_path, args.wordnet_path)

    with open(args.input) as f:
        lines = [line.strip() for line in f.readlines()]

    with open(args.output, "w") as writer:
        for i, line in enumerate(lines):
            aug_sentences = instance.generate_sentences(
                line,
                alpha_sr=args.alpha_sr,
                alpha_ri=args.alpha_ri,
                alpha_rs=args.alpha_rs,
                p_rd=args.alpha_rd,
                num_aug=args.num_aug,
            )
            for aug_sentence in aug_sentences:
                writer.write(aug_sentence + "\n")

    print(f"complete with output file: {args.output}")


if __name__ == "__main__":
    main()
