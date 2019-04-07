import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import struct
import numpy as np
import argparse
import pickle
import matplotlib.patches as patches
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
parser.add_argument('--true_path', type=str, default='.', help='file storing the path')
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

fig, ax = plt.subplots(1)

in_obs_file_name = in_folder+'obc'+str(args.env)+'.dat'
in_obs_file = open(in_obs_file_name, 'rb')
b_read = in_obs_file.read()
D = struct.unpack('d'*(args.N*args.dim), b_read)
D = np.array(D).reshape(shape)

plt.scatter(x=D[:,0], y=D[:,1], s=0.5)

def draw_robot(path, color):
    r = patches.Rectangle((path[0]-2/2,path[1]-5/2),2,5,linewidth=.5,edgecolor=color,facecolor='none')
    t = mpl.transforms.Affine2D().rotate_deg_around(path[0], path[1], path[2]) + ax.transData
    r.set_transform(t)
    ax.add_patch(r)
path = pickle.load( open(args.line_path, 'rb') )
for i in range(len(path)):
    draw_robot(path[i], 'r')

# ground truth
path = pickle.load( open(args.true_path, 'rb') )
for i in range(len(path)):
    draw_robot(path[i], 'b')
plt.show()
# Visualizer
#plt.imsave(out_folder+'obc'+str(i)+'.png', np.asarray(depth), dpi=1)
#vis.destroy_window()
in_obs_file.close()
