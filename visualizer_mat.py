import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import struct
import numpy as np
import argparse
import pickle
from data_loader_2d import *
print('--in_path: path to load point clouds')
print('--out_path: path to store screenshots')
print('--N: number of points in the point cloud')
print('--dim: dimension of point cloud')
print('--line_path: file storing the path information')
print('press enter to take screenshots')
parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default='../../../results/pt_cloud/', help='path to load point clouds')
parser.add_argument('--out_path', type=str, default='../../../results/screenshots/', help='path to store the screenshots')
parser.add_argument('--env_idx', type=int, default=0, help='env index')
parser.add_argument('--path_idx', type=int, default=0, help='path index')
parser.add_argument('--line_path', type=str, default='.', help='file storing the path')
#parser.add_argument('--true_path', type=str, default='.', help='file storing the path')
parser.add_argument('--n', type=int, default=3, help='number of attempts')
parser.add_argument('--dim', type=int, default=2, help='dimension of point cloud')
parser.add_argument('--encoding', type=str, default='')
parser.add_argument('--model', type=str, default='mpnet')
args = parser.parse_args()

in_path = args.in_path
out_path = args.out_path
#task_name = 'simple/'
in_folder = in_path
out_folder = out_path

fig, ax = plt.subplots(1)
test_data = load_test_dataset(N=1, NP=1, s=args.env_idx, sp=args.path_idx, folder=args.in_path)
obc, obs, paths, path_lengths = test_data
obc = obc[0]
shape=[[10.0,5.0],[5.0,10.0],[10.0,10.0],[10.0,5.0],[5.0,10.0],[10.0,5.0],[5.0,10.0]]
# add path information
# path is numpy array of dimension l*dim
# planned path
if args.encoding == 'latin1':
    path_sol = pickle.load( open(args.line_path+'mpnet_path_env_%d_path%d.p' % (args.env_idx, args.path_idx), 'rb'),encoding="latin1" )
else:
    path_sol = pickle.load( open(args.line_path, 'rb') )
paths = []
for i in range(args.n):
    if args.encoding == 'latin1':
        path = pickle.load( open(args.line_path+'mpnet_path_env_%d_path%d.p' % (args.env_idx, args.path_idx)+'_%d' % (i), 'rb'),encoding="latin1" )
    else:
        path = pickle.load( open(args.line_path+'mpnet_path_env_%d_path%d.p' % (args.env_idx, args.path_idx)+'_%d' % (i), 'rb') )
    paths.append(path)

alpha_max = .6
for p in range(len(paths)):
    path = paths[p]
    for i in range(len(path)-1):
        xmin = path[i][0]
        xmax = path[i+1][0]
        ymin = path[i][1]
        ymax = path[i+1][1]
        l = mlines.Line2D([xmin,xmax], [ymin,ymax], linewidth=0.5, color='b', alpha=alpha_max/(len(paths)+1)*(p+1))
        ax.add_line(l)
path = path_sol
for i in range(len(path)-1):
    xmin = path[i][0]
    xmax = path[i+1][0]
    ymin = path[i][1]
    ymax = path[i+1][1]
    l = mlines.Line2D([xmin,xmax], [ymin,ymax], linewidth=1., color='r')
    ax.add_line(l)

for i in range(0,7):
    r = patches.Rectangle((obc[i][0]-shape[i][0]/2,obc[i][1]-shape[i][1]/2),shape[i][0],shape[i][1],linewidth=.5,edgecolor='black',facecolor='black')
    ax.add_patch(r)

cs = 'gbrcyk'
mapping = np.random.randint(low=0, high=len(cs), size=len(path))
colors = [cs[mapping[i]] for i in range(len(path))]
xs = [path[i][0] for i in range(len(path))]
ys = [path[i][1] for i in range(len(path))]
plt.scatter(xs, ys, c=colors)
step = 5
ax.xaxis.set_ticks(np.arange(-20, 20+step, step))
ax.yaxis.set_ticks(np.arange(-20, 20+step, step))
#plt.show()
plt.savefig(args.out_path+'%s_path_env%d_path%d' % (args.model, args.env_idx,args.path_idx)+'.png')
