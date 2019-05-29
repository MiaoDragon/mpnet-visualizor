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
parser.add_argument('--planner', type=str, default='.', help='model name')
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
shape=[[10.0,5.0],[5.0,10.0],[10.0,10.0],[10.0,5.0],[5.0,10.0],[10.0,5.0],[5.0,10.0]]

in_path = args.in_path
out_path = args.out_path
#task_name = 'simple/'
in_folder = in_path
out_folder = out_path
D = []


# add path information
# path is numpy array of dimension l*dim
# planned path

graph = nx.read_graphml(args.tree_path+'%s_path_env%d_path%d.graphml' % (args.model, args.env_idx,args.path_idx))
if args.encoding == 'latin1':
    path_sol = pickle.load( open(args.tree_path+'%s_path_env%d_path%d.p' % (args.model, args.env_idx,args.path_idx), 'rb'),encoding="latin1" )
else:
    path_sol = pickle.load( open(args.tree_path+'%s_path_env%d_path%d.p' % (args.model, args.env_idx,args.path_idx), 'rb') )
edges = graph.edges_iter()

for (u, v) in edges:
    x = graph.node[u]['coords']
    y = graph.node[v]['coords']
    x = x.split(',')
    x = [float(i) for i in x]
    y = y.split(',')
    y = [float(j) for j in y]
    l = mlines.Line2D([x[0],y[0]], [x[1],y[1]], linewidth=1., color='pink',zorder=0,alpha=1.)
    ax.add_line(l)
nodes = graph.nodes_iter()
xs = []
ys = []
for u in nodes:
    x = graph.node[u]['coords']
    x = x.split(',')
    x = np.array([float(i) for i in x])
    xs.append(x[0])
    ys.append(x[1])
plt.scatter(xs, ys, s=.1, c='b', zorder=1, alpha=0.5)

path = path_sol
for i in range(len(path)-1):
    xmin = path[i][0]
    xmax = path[i+1][0]
    ymin = path[i][1]
    ymax = path[i+1][1]
    l = mlines.Line2D([xmin,xmax], [ymin,ymax], linewidth=2., color='r',zorder=1)
    ax.add_line(l)

for i in range(0,7):
    r = patches.Rectangle((obc[i][0]-shape[i][0]/2,obc[i][1]-shape[i][1]/2),shape[i][0],shape[i][1],\
        linewidth=.5,edgecolor=(126/255, 112/255, 125/255),facecolor=(126/255, 112/255, 125/255),zorder=1)
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
#cs = 'gbrcyk'
cs = 'gcyk'
mapping = np.random.randint(low=0, high=len(cs), size=len(path))
#colors = [cs[mapping[i]] for i in range(len(path))]
colors = ['darkblue', 'darkblue']
#xs = [path[i][0] for i in range(len(path))]
#ys = [path[i][1] for i in range(len(path))]
xs = [path[0][0],path[-1][0]]
ys = [path[0][1],path[-1][1]]
#sizes = [30. for i in range(len(path))]
#sizes[0] = 50.
#sizes[-1] = 50.
sizes = [100., 100.]
plt.scatter(xs, ys, s=sizes, c=colors, zorder=2)
step = 5
ax.xaxis.set_ticks(np.arange(-20, 20+step, step))
ax.yaxis.set_ticks(np.arange(-20, 20+step, step))
#fig.canvas.manager.window.wm_geometry("+%d+%d" % (center_x, center_y))
#plt.show()
plt.axis('off')
plt.savefig(args.out_path+'%s_path_env%d_path%d' % (args.model, args.env_idx,args.path_idx)+'.png', bbox_inches='tight',pad_inches=0)
