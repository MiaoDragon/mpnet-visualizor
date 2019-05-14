from ompl import base as ob
from ompl import geometric as og
from plan_s2d import IsInCollision
from data_loader_2d import *
import argparse
import pickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
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
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")


def getPathLengthObjective(si):
    return ob.PathLengthOptimizationObjective(si)

def plan(args):
    print('loading...')
    test_data = load_test_dataset(N=1, NP=1, s=args.env_idx, sp=args.path_idx, folder=args.data_path)
    obc, obs, paths, path_lengths = test_data
    obc = obc[0]
    s = paths[0][0][0].astype(np.float64)
    g = paths[0][0][path_lengths[0][0]-1].astype(np.float64)
    # create an SE2 state space
    space = ob.RealVectorStateSpace(2)
    # set lower anprint(start)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-20)
    bounds.setHigh(20)
    space.setBounds(bounds)
    # create a simple setup object
    start = ob.State(space)
    # we can pick a random start state...
    # ... or set specific values
    for i in range(len(s)):
        start[i] = s[i]
    goal = ob.State(space)
    for i in range(len(g)):
        goal[i] = g[i]
    si = ob.SpaceInformation(space)
    def isStateValid(state):
        return not IsInCollision(state, obc)
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    si.setup()
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)
    pdef.setOptimizationObjective(getPathLengthObjective(si))

    ss = allocatePlanner(si, args.planner)
    ss.setProblemDefinition(pdef)
    ss.setup()

    solved = ss.solve(5.0)
    path = pdef.getSolutionPath().getStates()
    solutions = np.zeros((len(path),2))
    for i in range(len(path)):
        solutions[i][0] = float(path[i][0])
        solutions[i][1] = float(path[i][1])
    path_file = os.path.join(args.model_path,'%s_path_env%d_path%d.p' % (args.planner, args.env_idx,args.path_idx))
    pickle.dump(solutions, open(path_file, "wb" ))

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../visual/')
parser.add_argument('--env_idx', type=int, default=10, help='which env to visualize?')
parser.add_argument('--path_idx', type=int, default=1253, help='which path to visualize?')
parser.add_argument('--data_path', type=str, default='../data/simple/')
parser.add_argument('--planner', type=str, default='bitstar')
args = parser.parse_args()
plan(args)
