import argparse
import warnings as w

from emp.opt import Optimizer

w.filterwarnings('ignore')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HOUSE', help='choose patrolling environment')
    parser.add_argument('--size_MRS', type=int, default=2, help='adjust group size of MRS')
    parser.add_argument('--epoch', type=int, default=2000, help='set the number of epochs of optimization')
    parser.add_argument('--step', type=int, default=100000, help='set the number of steps of execution')
    parser.add_argument('--lr', type=float, default=1e-4, help='adjust learning rate of ADAM optimizer of policies')
    parser.add_argument('--epsilon', type=float, default=1e-12, help='adjust epsilon for mask smoothing and NaN avoiding')
    parser.add_argument('--verbose', action='store_true', help='print records of optimization')
    parser.add_argument('--output', action='store_true', help='output optimization history and the best policies')

    parser.add_argument('--alpha_r', type=float, default=0.0, help='set weight for robustness (average entropy)')
    parser.add_argument('--alpha_s', type=float, default=0.0, help='set weight for unpredictability (average entropy rate)')
    parser.add_argument('--variational', action='store_true', help='replace entropy maxi. with KLD mini. to user-defined distribution')
    operation = parser.parse_args()

    return operation


def main(operation):
    EMP = Optimizer(operation)
    EMP.trainPolicy(operation)
    EMP.testPolicy(operation)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
