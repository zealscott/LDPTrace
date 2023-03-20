import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epsilon', type=float, default=1.0,
                    help='Privacy budget')
parser.add_argument('--grid_num', type=int, default=6,
                    help='Number of grids is n x n')
parser.add_argument('--query_num', type=int, default=200,
                    help='Number of experiment queries')
parser.add_argument('--dataset', type=str, default='oldenburg')
parser.add_argument('--re_syn', action='store_true',
                    help='Synthesizing or use existing file')
parser.add_argument('--max_len', type=float, default=0.9,
                    help='Quantile of estimated max length')
parser.add_argument('--size_factor', type=float, default=9.0,
                    help='Quantile of estimated max length')
parser.add_argument('--multiprocessing', action='store_true')


args = parser.parse_args()
