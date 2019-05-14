import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import struct
import numpy as np
import argparse
import pickle
import networkx as nx
from data_loader_2d import *
print('--in_path: path to load point clouds')
print('--out_path: path to store screenshots')
print('--N: number of points in the point cloud')
print('--dim: dimension of point cloud')
print('--tree_path: file storing the path information')
print('press enter to take screenshots')
parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default='../../../results/pt_cloud/', help='path to load point clouds')
parser.add_argument('--out_path', type=str, default='../../../results/screenshots/', help='path to store the screenshots')
parser.add_argument('--env_idx', type=int, default=0, help='env index')
parser.add_argument('--path_idx', type=int, default=0, help='path index')
parser.add_argument('--tree_path', type=str, default='.', help='file storing the path')
#parser.add_argument('--true_path', type=str, default='.', help='file storing the path')
parser.add_argument('--N', type=int, default=5, help='number of points in the point cloud')
parser.add_argument('--dim', type=int, default=2, help='dimension of point cloud')
parser.add_argument('--encoding', type=str, default='')
parser.add_argument('--model', type=str, default='mpnet')
args = parser.parse_args()


fig, ax = plt.subplots(1)

test_data = load_test_dataset(N=1, NP=1, s=args.env_idx, sp=args.path_idx, folder=args.in_path)
obc, obs, paths, path_lengths = test_data
obc = obc[0]
size = 5.

shape = (args.N, args.dim)
in_path = args.in_path
out_path = args.out_path
#task_name = 'simple/'
in_folder = in_path
out_folder = out_path
D = []


# add path information
# path is numpy array of dimension l*dim
# planned path
graph = nx.read_graphml(args.tree_path)
edges = graph.edges_iter()

for (u, v) in edges:
    x = graph.node[u]['coords']
    y = graph.node[v]['coords']
    x = x.split(',')
    x = [float(i) for i in x]
    y = y.split(',')
    y = [float(j) for j in y]
    l = mlines.Line2D([x[0],y[0]], [x[1],y[1]], linewidth=0.5, color='r')
    ax.add_line(l)

for i in range(0,7):
    r = patches.Rectangle((obc[i][0]-size/2,obc[i][1]-size/2),size,size,linewidth=.5,edgecolor='black',facecolor='black')
    ax.add_patch(r)

# ground truth
"""
path = pickle.load( open(args.true_path, 'rb') )
if args.dim == 2:
    path = np.concatenate((path, np.zeros((path.shape[0],1))), axis=1)
ground_line_set = LineSet()
ground_line_set.points = Vector3dVector(path)
lines = [[i,i+1] for i in range(len(path.shape[0])-1)]
ground_line_set.lines = Vector2iVector(lines)
colors = [[0, 1, 0] for i in range(len(lines))]
ground_line_set.colors = Vector3dVector(colors)
"""
step = 4
ax.xaxis.set_ticks(np.arange(-20, 20+step, step))
ax.yaxis.set_ticks(np.arange(-20, 20+step, step))
#fig.canvas.manager.window.wm_geometry("+%d+%d" % (center_x, center_y))
#plt.show()
plt.savefig(args.out_path+args.model+'.png')
