import torch
import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pdb
from scipy.spatial.transform import Rotation as R
import pyquaternion as pyquat
import seaborn as sns

def read_cameras(meta, H, W):
    K = np.array(meta["camera"]["K"]) # 3x3
    K[0] *= W # multiplying first row by W
    K[1] *= H # multiplying second row by H

    poses = []
    for i in range(len(meta["camera"]["positions"])):
        pose = np.eye(4)
        t = np.array(meta["camera"]["positions"][i])
        q = np.array(meta["camera"]["quaternions"][i])
        rot = pyquat.Quaternion(*q).rotation_matrix
        pose[:3, :3] = rot
        pose[:3, 3] = t
        poses.append(pose)
    return K, poses

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.grid(True)
        self.ax.tick_params(axis='both', which='major', labelsize=10)
        sns.set(style='whitegrid')
        # # turn of axes
        # self.ax.set_axis_off()
        # self.ax.set_xticks([])
        # self.ax.set_yticks([])
        # self.ax.set_zticks([])
        # self.ax.set_xticklabels([])
        # self.ax.set_yticklabels([])
        # self.ax.set_zticklabels([])
        # self.ax.set_aspect('equal')
        print('initialize camera pose visualizer')

    def set_alpha_values(self, poses):
        t_arr = np.array([pose[:3, -1] for pose in poses])
        # alphas for cameras in back should be smaller
        # coordinate of the frontmost camera:
        t_front = t_arr[np.argmax(t_arr[:, 1])]
        # coordinate of the backmost camera:
        t_back = t_arr[np.argmin(t_arr[:, 1])]
        # distance between frontmost and backmost camera:
        dist = np.linalg.norm(t_front - t_back)
        # alphas for cameras in back should be smaller
        # 0.1 to 0.35
        alphas = ((t_arr[:, 1] - t_back[1]) / dist) * 0.25 + 0.1
        return [alphas[i] for i in range(len(poses))]

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3, alpha=0.35):
        focal_len_scaled = -1 * focal_len_scaled
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled,
                                1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled,
                                1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled,
                                1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                  [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1],
                   vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=alpha))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()

if __name__ == '__main__':
    ### Blender Lego loader
    # poses = []
    # data = json.load(open('/work/yashsb/datasets/NeRF_datasets/nerf_synthetic/lego/transforms_train.json'))
    # for frame in data['frames']:
    #     poses.append(np.array(frame['transform_matrix']))
    ############################

    ### Many Object Scenes loader
    meta = json.load(open('/work/yashsb/panoptic-lifting/data/many_object_scenes/large_corridor_2/time_0/metadata.json'))
    _, poses = read_cameras(meta, 512, 512)
    # subsample poses which is a list
    poses = poses[::4]
    ############################

    # ### ScanNet loader
    # pose_dir = "/work/yashsb/panoptic-lifting/data/scannet/scene0050_02/pose"
    # poses = []
    # opencv2blender = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # # read all txt files in pose_dir
    # for file in os.listdir(pose_dir):
    #     if file.endswith(".txt"):
    #         pose = np.loadtxt(os.path.join(pose_dir, file))
    #         pose = pose @ opencv2blender
    #         poses.append(pose)
    # poses = poses[::15]
    # ############################

    t_arr = np.array([pose[:3,-1] for pose in poses])
    maxes = t_arr.max(axis=0)
    mins = t_arr.min(axis=0)
    mean_t = t_arr.mean(axis=0)

    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([mins[0]-1, maxes[0]+1], [mins[1]-1, maxes[1]+1], [mins[2]-1, maxes[2]+1])
    alphas = visualizer.set_alpha_values(poses)

    # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
    for pose, alpha in zip(poses, alphas):
        visualizer.extrinsic2pyramid(pose, 'limegreen', 1.0, aspect_ratio=0.35, alpha=alpha)

    plt.savefig('camera_poses.pdf', format='pdf', dpi=300, bbox_inches='tight')
    visualizer.show()
