from Model.GEM_end2end_model import End2EndMPNet
#from GEM_end2end_model_rand import End2EndMPNet as End2EndMPNet_rand
import Model.model as model
import Model.model_c2d as model_c2d
import Model.AE.CAE_r3d as CAE_r3d
import Model.AE.CAE as CAE_2d
import torch
from torch.autograd import Variable
from ompl import base as ob
from ompl import geometric as og
import plan_c2d, plan_s2d, plan_r2d, plan_r3d
import data_loader_2d, data_loader_r2d, data_loader_r3d
import argparse
import pickle
import sys
import time
import os
import numpy as np
from plan_general import *
from utility import *
import utility_s2d, utility_c2d, utility_r3d, utility_r2d
import random
def eval_tasks(mpNet, test_data, env_idx, path_idx, IsInCollision, normalize_func = lambda x:x, unnormalize_func=lambda x: x, time_flag=False):
    obc, obs, paths, path_lengths = test_data
    obs = torch.from_numpy(obs)
    fes_env = []   # list of list
    valid_env = []
    #for i in range(0,1):
    time_env = []
    time_total = []
    path = None
    path_attempts = []
    for i in range(env_idx, env_idx+1):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        # feasible paths for each env
        for j in range(path_idx, path_idx+1):
            time0 = time.time()
            time_norm = 0.
            fp = 0 # indicator for feasibility
            print ("step: i="+str(i)+" j="+str(j))
            #for j in range(0,2):
            p1_ind=0
            p2_ind=0
            p_ind=0
            if path_lengths[i][j]==0:
                # invalid, feasible = 0, and path count = 0
                fp = 0
                valid_path.append(0)
            if path_lengths[i][j]>0:
                fp = 0
                valid_path.append(1)
                path = [torch.from_numpy(paths[i][j][0]).type(torch.FloatTensor),\
                        torch.from_numpy(paths[i][j][path_lengths[i][j]-1]).type(torch.FloatTensor)]
                step_sz = DEFAULT_STEP
                MAX_NEURAL_REPLAN = 11
                for t in range(MAX_NEURAL_REPLAN):
                # adaptive step size on replanning attempts
                    if (t == 2):
                        step_sz = 1.2
                    elif (t == 3):
                        step_sz = 0.5
                    elif (t > 3):
                        step_sz = 0.1
                    if time_flag:
                        path, time_norm = neural_replan(mpNet, path, obc[i], obs[i], IsInCollision, \
                                            normalize_func, unnormalize_func, t==0, step_sz=step_sz, time_flag=time_flag)
                    else:
                        path = neural_replan(mpNet, path, obc[i], obs[i], IsInCollision, \
                                            normalize_func, unnormalize_func, t==0, step_sz=step_sz, time_flag=time_flag)
                    path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                    path_attempts.append(path)
                    if feasibility_check(path, obc[i], IsInCollision, step_sz=0.01):
                        fp = 1
                        print('feasible, ok!')
                        break
        if fp:
            time1 = time.time() - time0
            print('test time: %f' % (time1))
            # calculate path length
            path_length = 0.
            #path = path.numpy()
            for i in range(len(path)-1):
                path_length += np.linalg.norm(path[i+1]-path[i])
            return True, time1, path_length, path, path_attempts
        else:
            return False, 0., 0., None, []

def useGraphTool(pd):
    graphml = pd.printGraphML()
    f = open("graph.graphml", 'w')
    f.write(graphml)
    f.close()

def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        planner = og.BITstar(si)
        planner.setPruning(False)
        return planner
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    elif plannerType.lower() == 'rrtconnect':
        return og.RRTConnect(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")


def getPathLengthObjective(si, length):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(length))
    return obj
def plan(args):
    print('loading...')
    if args.env_type == 's2d':
        data_loader = data_loader_2d
        IsInCollision = plan_s2d.IsInCollision
        # create an SE2 state space
        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-20)
        bounds.setHigh(20)
        space.setBounds(bounds)
        time_limit = 20.
        normalize = utility_s2d.normalize
        unnormalize = utility_s2d.unnormalize
        CAE = CAE_2d
        MLP = model.MLP
    elif args.env_type == 'c2d':
        data_loader = data_loader_2d
        IsInCollision = plan_c2d.IsInCollision
        # create an SE2 state space
        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-20)
        bounds.setHigh(20)
        space.setBounds(bounds)
        time_limit = 20.
        normalize = utility_c2d.normalize
        unnormalize = utility_c2d.unnormalize
        CAE = CAE_2d
        MLP = model_c2d.MLP
    elif args.env_type == 'r2d':
        data_loader = data_loader_r2d
        IsInCollision = plan_r2d.IsInCollision
        # create an SE2 state space
        space = ob.SE2StateSpace()
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-20)
        bounds.setHigh(20)
        space.setBounds(bounds)
        time_limit = 50.
    elif args.env_type == 'r3d':
        data_loader = data_loader_r3d
        IsInCollision = plan_r3d.IsInCollision
        # create an SE2 state space
        space = ob.RealVectorStateSpace(3)
        bounds = ob.RealVectorBounds(3)
        bounds.setLow(-20)
        bounds.setHigh(20)
        space.setBounds(bounds)
        time_limit = 20.
    mpNet = End2EndMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, \
                args.output_size, 'deep', args.n_tasks, args.n_memories, args.memory_strength, args.grad_step, \
                CAE, MLP)
    model_path='cmpnet_epoch_%d.pkl' %(args.start_epoch)
    if args.start_epoch > 0:
        load_net_state(mpNet, os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
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

    test_data = data_loader.load_test_dataset(N=args.N, NP=args.NP, s=args.env_idx, sp=args.path_idx, folder=args.data_path)
    obcs, obs, paths, path_lengths = test_data
    time_env = []
    time_total = []
    fes_env = []   # list of list
    valid_env = []
    unnormalize_func=lambda x: unnormalize(x, 20.)

    for i in range(len(paths)):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        # feasible paths for each env
        obc = obcs[i]
        for j in range(len(paths[0])):

            s = paths[i][j][0].astype(np.float64)
            g = paths[i][j][path_lengths[i][j]-1].astype(np.float64)
            time0 = time.time()
            fp = 0 # indicator for feasibility
            print ("step: i="+str(i)+" j="+str(j))
            #for j in range(0,2):
            p1_ind=0
            p2_ind=0
            p_ind=0
            data_length = 0.
            for k in range(path_lengths[i][j]-1):
                data_length += np.linalg.norm(paths[i][j][k+1]-paths[i][j][k])
            print('data length:')
            print(data_length)
            fes, mp_time, mp_length, mpnet_path, path_attempts = eval_tasks(mpNet, test_data, i, j, IsInCollision, unnormalize_func)
            if not fes:
                continue
            time_limit = 10
            # if mpnet path length is within 110% of data length
            if mp_length > data_length * 1.05:
                continue
            data_length = mp_length
            print('mpnet length:')
            print(data_length)
            if path_lengths[i][j]==0:
                # invalid, feasible = 0, and path count = 0
                fp = 0
                valid_path.append(0)
            if path_lengths[i][j]>0:
                fp = 0
                valid_path.append(1)
                # create a simple setup object
                start = ob.State(space)
                # we can pick a random start state...
                # ... or set specific values
                for k in range(len(s)):
                    start[k] = s[k]
                goal = ob.State(space)
                for k in range(len(g)):
                    goal[k] = g[k]
                si = ob.SpaceInformation(space)
                def isStateValid(state):
                    return not IsInCollision(state, obc)
                si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
                si.setup()
                pdef = ob.ProblemDefinition(si)
                pdef.setStartAndGoalStates(start, goal)
                pdef.setOptimizationObjective(getPathLengthObjective(si, data_length))
                planner_paths = []
                planner_graphs = []
                planners = ['bitstar', 'informedrrtstar', 'rrtstar', 'rrtconnect']
                for planner in planners:
                    fp = 0
                    ss = allocatePlanner(si, planner)
                    ss.setProblemDefinition(pdef)
                    ss.setup()

                    solved = ss.solve(time_limit)
                    if solved:
                        fp = 1
                        # compare path length
                        ompl_path = ob.PlannerData(si)
                        #ss.getPlannerData(path)
                        #path_file = os.path.join(args.model_path,'%s_path_env%d_path%d.graphml' % (args.planner, args.env_idx,args.path_idx))
                        graphml = ompl_path.printGraphML()
                        #f = open(path_file, 'w')
                        #f.write(graphml)
                        #f.close()
                        ompl_path = pdef.getSolutionPath().getStates()
                        solutions = np.zeros((len(ompl_path),2))
                        for i in range(len(ompl_path)):
                            solutions[i][0] = float(ompl_path[i][0])
                            solutions[i][1] = float(ompl_path[i][1])
                        plan_length = 0.
                        for i in range(len(ompl_path)-1):
                            plan_length += np.linalg.norm(solutions[i+1] - solutions[i])
                        if plan_length < .9 * data_length:
                            # found a better path
                            print('Ooooops, MPNet is worse')
                            fp = 0
                        if fp:
                            planner_paths.append(solutions)
                            planner_graphs.append(graphml)
                        #pickle.dump(solutions, open(args.model_path+'%s_path_env%d_path%d.p' % (args.planner, args.env_idx, args.path_idx), 'wb'))
                    else:
                        fp = 0
                    if not fp:
                        break
                if not fp:
                    continue
                print('found one path!')
                path = np.array([p.numpy() for p in path])
                pickle.dump(path, open(path_folder+'mpnet_path_env_%d_path%d.p' % (args.env_idx+i, args.path_idx+j), "wb" ))
                for path_attempt_i in range(len(path_attempts)):
                    path_attempt = path_attempts[path_attempt_i]
                    pickle.dump(path_attempt, open(path_folder+'mpnet_path_env_%d_path%d.p_%d' % (args.env_idx+i, args.path_idx+j, path_attempt_i), 'wb'))

                for planner_i in range(len(planners)):
                    planner = planners[planner_i]
                    graphml = planner_graphs[planner_i]
                    path = planner_paths[planner_i]
                    path_file = os.path.join(args.model_path,'%s_path_env%d_path%d.graphml' % (planner, args.env_idx+i,args.path_idx+j))
                    f = open(path_file, 'w')
                    f.write(graphml)
                    f.close()
                    pickle.dump(solutions, open(args.model_path+'%s_path_env%d_path%d.p' % (planner, args.env_idx+i, args.path_idx+j), 'wb'))

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../visual/')
parser.add_argument('--env_idx', type=int, default=10, help='which env to visualize?')
parser.add_argument('--path_idx', type=int, default=1253, help='which path to visualize?')
parser.add_argument('--N', type=int, default=0)
parser.add_argument('--NP', type=int, default=0)

parser.add_argument('--data_path', type=str, default='../data/simple/')
parser.add_argument('--env_type', type=str, default='s2d')
#parser.add_argument('--planner', type=str, default='bitstar')
parser.add_argument('--data_type', type=str, default='seen')


parser.add_argument('--n_tasks', type=int, default=1,help='number of tasks')
parser.add_argument('--n_memories', type=int, default=256, help='number of memories for each task')
parser.add_argument('--memory_strength', type=float, default=0.5, help='memory strength (meaning depends on memory)')
parser.add_argument('--total_input_size', type=int, default=2800+4, help='dimension of total input')
parser.add_argument('--AE_input_size', type=int, default=2800, help='dimension of input to AE')
parser.add_argument('--mlp_input_size', type=int , default=28+4, help='dimension of the input vector')
parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--device', type=int, default=0, help='cuda device')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--memory_type', type=str, default='res', help='res for reservoid, rand for random sampling')
parser.add_argument('--world_size', type=int, default=50, help='boundary of world')
parser.add_argument('--grad_step', type=int, default=1, help='number of gradient steps in continual learning')
args = parser.parse_args()
plan(args)
