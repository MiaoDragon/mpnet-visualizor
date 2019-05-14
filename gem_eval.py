import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable
import math
import time
from plan_general import *
import copy

def eval_tasks(mpNet, test_data, true_file, path_folder, env_idx, path_idx, IsInCollision, normalize_func = lambda x:x, unnormalize_func=lambda x: x):
    obc, obs, paths, path_lengths = test_data
    obs = torch.from_numpy(obs)
    fes_env = []   # list of list
    valid_env = []
    #for i in range(0,1):
    time_env = []
    time_total = []
    path = None
    for i in range(len(paths)):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        # feasible paths for each env
        for j in range(len(paths[0])):
            path_attempts = []
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
                print('path length is 0')
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
                        step_sz = 2.0
                    elif (t == 3):
                        step_sz = 1.2
                    elif (t > 3):
                        step_sz = 0.5
                    path = neural_replan2(mpNet, path, obc[i], obs[i], IsInCollision, \
                                         normalize_func, unnormalize_func, t==0, step_sz=step_sz)
                    #print('returned path:')
                    #print(path)
                    path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                    #print('after lvc:')
                    #print(path)
                    path_attempts.append(copy.deepcopy(path))
                    #pickle.dump(path, open(path_file+'_'+str(t), "wb" ))
                    if feasibility_check(path, obc[i], IsInCollision, step_sz=0.01):
                        fp = 1
                        print('feasible, ok!')
                        break
            #if path is None:
            #    return 0
            #for p in path:
            #    IsInCollision(p.numpy(), obc[i], True)
            if len(path) > 5:
                num_saved_path += 1
                path = np.array([p.numpy() for p in path])
                pickle.dump(path, open(path_folder+'path_env_%d_path%d.p' % (env_idx+i, path_idx+j), "wb" ))
                for path_attempt_i in range(len(path_attempts)):
                    path_attempt = path_attempts[path_attempt_i]
                    pickle.dump(path_attempt, open(path_folder+'path_env_%d_path%d.p_%d' % (env_idx+i, path_idx+j, path_attempt_i), 'wb'))
                if num_saved_path > 10:
                    return 1
    #path = np.array(paths[i][j])
    #pickle.dump(path, open(true_file, 'wb'))
    # write as txt files
    #file = open(filename, 'w')
    #file.write('planned path:')
    #for p in path:
    #    file.write('(' + ','.join(p) + ')\n')

    #file.write('data:')
    #for i in range(len(path_lengths[0][0])):
    #    file.write('(' + ','.join(paths[0][0][i]) + ')\n')
    return 1
