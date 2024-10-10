import argparse
from operator import add

def get_args_parser():
    parser = argparse.ArgumentParser("a job for submit", add_help=False)
    parser.add_argument("--a", default=5, type=int)
    parser.add_argument("--b", default=7, type=int)
    return parser

def main(args):
    import os

    cmd = 'nvidia-smi'
    os.system(cmd)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)