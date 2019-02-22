from __future__ import print_function
from Model.GEM_end2end_model import End2EndMPNet
import numpy as np
import argparse
import os
import torch
from gem_eval import eval_tasks
import plan_s2d, plan_c2d
import data_loader
from torch.autograd import Variable
import copy
import os
import random
from utility import *

def main(args):
    # set seed
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    if args.memory_type == 'res':
        mpNet = End2EndMPNet(args.mlp_input_size, args.output_size, 'deep', \
                    args.n_tasks, args.n_memories, args.memory_strength, args.grad_step)
    elif args.memory_type == 'rand':
        #mpNet = End2EndMPNet_rand(args.mlp_input_size, args.output_size, 'deep', \
        #            args.n_tasks, args.n_memories, args.memory_strength, args.grad_step)
        pass
    # load previously trained model if start epoch > 0
    model_path = args.model_path + args.model_name
    load_net_state(mpNet, model_path)
    # set seed after loading
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)

    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()
        mpNet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
    if args.start_epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))
    # setup evaluation function, and load function
    if args.env_type == 's2d':
        IsInCollision = plan_s2d.IsInCollision
        load_test_dataset = data_loader.load_test_dataset
    elif args.env_type == 'c2d':
        IsInCollision = plan_c2d.IsInCollision
        load_test_dataset = data_loader.load_test_dataset
    # load train and test data
    print('loading...')
    test_data = load_test_dataset(N=0, NP=0, s=args.env_idx, sp=args.path_idx, folder=args.data_path)
    # test
    # testing
    print('testing...')
    # unnormalize function
    unnormalize_func=lambda x: unnormalize(x, args.world_size)
    path_file = os.path.join(args.model_path,'path_env%d_path%d.p' % (args.env_idx,args.path_idx))
    eval_tasks(mpNet, test_data, path_file, IsInCollision, unnormalize_func)

parser = argparse.ArgumentParser()
# for training
parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
parser.add_argument('--model_name', type=str, default='cmpnet', help='filename of model')
# for continual learning
parser.add_argument('--n_tasks', type=int, default=1,help='number of tasks')
parser.add_argument('--n_memories', type=int, default=256, help='number of memories for each task')
parser.add_argument('--memory_strength', type=float, default=0.5, help='memory strength (meaning depends on memory)')
# Model parameters
parser.add_argument('--mlp_input_size', type=int , default=28+4, help='dimension of the input vector')
parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--device', type=int, default=0, help='cuda device')
parser.add_argument('--data_path', type=str, default='../data/simple/')
parser.add_argument('--memory_type', type=str, default='res', help='res for reservoid, rand for random sampling')
parser.add_argument('--env_type', type=str, default='s2d', help='s2d for simple 2d, c2d for complex 2d')
parser.add_argument('--world_size', type=int, default=50, help='boundary of world')
parser.add_argument('--env_idx', type=int, default=0, help='which env to visualize?')
parser.add_argument('--path_idx', type=int, default=0, help='which path to visualize?')
args = parser.parse_args()
print(args)
main(args)
