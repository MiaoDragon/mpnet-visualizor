import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import struct
import numpy as np
import argparse
from open3d import *
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
parser.add_argument('--N', type=int, default=5, help='number of points in the point cloud')
parser.add_argument('--dim', type=int, default=2, help='dimension of point cloud')
args = parser.parse_args()

shape = (args.N, args.dim)
in_path = args.in_path
out_path = args.out_path
#task_name = 'simple/'
in_folder = in_path
out_folder = out_path
D = []

in_obs_file_name = in_folder+'obc'+str(i)+'.dat'
in_obs_file = open(in_obs_file_name, 'rb')
b_read = in_obs_file.read()
D = struct.unpack('d'*(args.N*args.dim), b_read)
D = np.array(D).reshape(shape)
# append zero to z dimension if 2 dim
if args.dim == 2:
    D = np.concatenate((D, np.zeros((shape[0],1))), axis=1)
u = np.mean(D, axis=0)
pcd = PointCloud()
pcd.points = Vector3dVector(D)
# add path information
# path is numpy array of dimension l*dim
path = pickle.load( open(args.line_path, 'rb') )
if args.dim == 2:
    path = np.concatenate((path, np.zeros((path.shape[0],1))), axis=1)
line_set = LineSet()
line_set.points = Vector3dVector(path)
lines = [[i,i+1] for i in range(len(path.shape[0])-1)]
line_set.lines = Vector2iVector(lines)
colors = [[1, 0, 0] for i in range(len(lines))]
line_set.colors = Vector3dVector(colors)
# Visualizer
vis = Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.add_geometry(line_set)
vis.run()
depth = vis.capture_screen_float_buffer()
plt.imsave(out_folder+'obc'+str(i)+'.png', np.asarray(depth), dpi=1)
vis.destroy_window()
in_obs_file.close()
