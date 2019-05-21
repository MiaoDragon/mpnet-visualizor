from ompl import base as ob
from ompl import geometric as og
import plan_c2d, plan_s2d, plan_r2d, plan_r3d
import data_loader_2d, data_loader_r2d, data_loader_r3d, data_loader_rigid
import argparse
import pickle
import sys
import time
import os
import numpy as np
def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
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

    test_data = data_loader.load_test_dataset(N=args.N, NP=args.NP, s=args.env_idx, sp=args.path_idx, folder=args.data_path)
    obcs, obs, paths, path_lengths = test_data
    time_env = []
    time_total = []
    fes_env = []   # list of list
    valid_env = []
    for i in range(len(paths)):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        # feasible paths for each env
        obc = obcs[i]
        for j in range(len(paths[0])):
            data_length = 0.
            for k in range(path_lengths[i][j]-1):
                data_length += np.linalg.norm(paths[i][j][k+1]-paths[i][j][k])
            s = paths[i][j][0].astype(np.float64)
            g = paths[i][j][path_lengths[i][j]-1].astype(np.float64)
            time0 = time.time()
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

                ss = allocatePlanner(si, args.planner)
                ss.setProblemDefinition(pdef)
                ss.setup()

                solved = ss.solve(time_limit)
                if solved:
                    fp = 1
            if fp:
                # only for successful paths
                time1 = time.time() - time0
                time_path.append(time1)
                print('test time: %f' % (time1))
            fes_path.append(fp)
        time_env.append(time_path)
        time_total += time_path
        print('average test time up to now: %f' % (np.mean(time_total)))
        fes_env.append(fes_path)
        valid_env.append(valid_path)
        print('accuracy up to now: %f' % (np.sum(fes_env) / np.sum(valid_env)))
    pickle.dump(time_env, open(args.model_path+'time_%s.p' % (args.data_type), "wb" ))
    f = open(os.path.join(args.model_path,'%s_accuracy.txt' % (args.data_type)), 'w')
    valid_env = np.array(valid_env).flatten()
    fes_env = np.array(fes_env).flatten()   # notice different environments are involved
    seen_test_suc_rate = fes_env.sum() / valid_env.sum()
    f.write(str(seen_test_suc_rate))
    f.close()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../visual/')
parser.add_argument('--env_idx', type=int, default=10, help='which env to visualize?')
parser.add_argument('--path_idx', type=int, default=1253, help='which path to visualize?')
parser.add_argument('--N', type=int, default=0)
parser.add_argument('--NP', type=int, default=0)

parser.add_argument('--data_path', type=str, default='../data/simple/')
parser.add_argument('--env_type', type=str, default='s2d')
parser.add_argument('--planner', type=str, default='bitstar')
parser.add_argument('--data_type', type=str, default='seen')
args = parser.parse_args()
plan(args)
