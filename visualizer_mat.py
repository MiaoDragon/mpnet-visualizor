import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import struct
import numpy as np
import argparse
import pickle
print('--in_path: path to load point clouds')
print('--out_path: path to store screenshots')
print('--N: number of points in the point cloud')
print('--dim: dimension of point cloud')
print('--line_path: file storing the path information')
print('press enter to take screenshots')
parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default='../../../results/pt_cloud/', help='path to load point clouds')
parser.add_argument('--out_path', type=str, default='../../../results/screenshots/', help='path to store the screenshots')
parser.add_argument('--env', type=int, default=0, help='env index')
parser.add_argument('--path_i', type=int, default=0, help='path index')
parser.add_argument('--line_path', type=str, default='.', help='file storing the path')
#parser.add_argument('--true_path', type=str, default='.', help='file storing the path')
parser.add_argument('--N', type=int, default=5, help='number of points in the point cloud')
parser.add_argument('--dim', type=int, default=2, help='dimension of point cloud')
parser.add_argument('--encoding', type=str, default='')
parser.add_argument('--model', type=str, default='mpnet')
args = parser.parse_args()

shape = (args.N, args.dim)
in_path = args.in_path
out_path = args.out_path
#task_name = 'simple/'
in_folder = in_path
out_folder = out_path
D = []

fig, ax = plt.subplots(1)
in_obs_file_name = in_folder+'obc'+str(args.env)+'.dat'
in_obs_file = open(in_obs_file_name, 'rb')
b_read = in_obs_file.read()
D = struct.unpack('d'*(args.N*args.dim), b_read)
D = np.array(D).reshape(shape)
# append zero to z dimension if 2 dim
plt.scatter(x=D[:,0], y=D[:,1], s=0.5)
# add path information
# path is numpy array of dimension l*dim
# planned path
if args.encoding == 'latin1':
    path = pickle.load( open(args.line_path, 'rb'),encoding="latin1" )
else:
    path = pickle.load( open(args.line_path, 'rb') )
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
for i in range(path.shape[0]-1):
    xmin = path[i][0]
    xmax = path[i+1][0]
    ymin = path[i][1]
    ymax = path[i+1][1]
    l = mlines.Line2D([xmin,xmax], [ymin,ymax], linewidth=0.5, color='r')
    ax.add_line(l)

#plt.show()
plt.savefig(args.out_path+args.model+'.png')
in_obs_file.close()
