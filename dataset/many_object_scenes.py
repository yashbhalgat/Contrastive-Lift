# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import random
from pathlib import Path

import torch
import pickle
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import pyquaternion as pyquat

from dataset.base import create_segmentation_data_base, BaseDataset, process_bounding_box_dict
from dataset.preprocessing.preprocess_scannet import get_thing_semantics
from util.camera import compute_world2normscene
from util.misc import EasyDict
from util.ray import get_ray_directions_with_intrinsics, get_rays, rays_intersect_sphere


def read_cameras(meta, H, W):
    K = np.array(meta["camera"]["K"]) # 3x3
    K[0] *= W # multiplying first row by W
    K[1] *= H # multiplying second row by H
    K = np.abs(K) # a bit hacky... :/

    poses = []
    for i in range(len(meta["camera"]["positions"])):
        pose = np.eye(4)
        t = np.array(meta["camera"]["positions"][i])
        q = np.array(meta["camera"]["quaternions"][i])
        rot = pyquat.Quaternion(*q).rotation_matrix
        pose[:3, :3] = rot
        pose[:3, 3] = t
        # # we may need to convert blender convention to opencv convention
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pose = pose @ blender2opencv
        poses.append(pose)
    return K, poses


class MOSDataset(BaseDataset): # many-object-scenes dataset

    def __init__(self, root_dir, split, image_dim, max_depth, overfit=False, num_val_samples=8, load_depth=False, load_feat=False, feature_type="nearest",
                 instance_dir='detic_instance', semantics_dir="detic_semantic",
                 instance_to_semantic_key='instance_to_semantic', create_seg_data_func=create_segmentation_data_base, subsample_frames=1,
                 do_not_load=False): # do_not_load is used when searching optimal bandwidth. We want to use train data but don't want to load rays, etc.
        super().__init__(root_dir, split, image_dim, max_depth, overfit, num_val_samples, load_depth, load_feat, instance_dir, semantics_dir, instance_to_semantic_key, create_seg_data_func, subsample_frames, False, do_not_load)
        self.faulty_classes = [0]
        self.is_thing = get_thing_semantics()
        self.all_frame_names = []
        self.all_probabilities, self.all_confidences = [], []
        self.all_origins = []
        self.all_feats = []
        self.world2scene = np.eye(4, dtype=np.float32)
        self.force_reset_fov = False
        self.full_train_set_mode = True
        self.random_train_val_ratio = 0.8 # 20% of frames are used for validation
        self.num_semantics = 2 # Foreground and Background
        self.do_not_load = do_not_load
        self.feature_type = feature_type
        self.setup_data()

    def setup_data(self):
        self.all_frame_names = sorted([x.stem for x in (self.root_dir / "color").iterdir() if x.name.endswith('.png')], key=lambda y: int(y) if y.isnumeric() else y)
        sample_indices = list(range(len(self.all_frame_names)))
        
        # use self.random_train_val_ratio to select last 20% as test set
        # this works because the frames were generated at random to begin with
        # also, always using last 20% means the test set is deterministic and fixed for all experiments
        self.val_indices = sample_indices[int(len(self.all_frame_names) * self.random_train_val_ratio):]
        self.train_indices = [sample_index for sample_index in sample_indices if sample_index not in self.val_indices]
        
        self.train_indices = self.train_indices[::self.subsample_frames]
        self.val_indices = self.val_indices[::self.subsample_frames]
        dims, intrinsics, cam2scene = [], [], []
        img_h, img_w = np.array(Image.open(self.root_dir / "color" / f"{self.all_frame_names[0]}.png")).shape[:2]
        metadata = json.load(open(self.root_dir / "metadata.json"))
        K, camera2world_list = read_cameras(metadata, img_h, img_w)
        for sample_index in sample_indices:
            intrinsic_color = K.copy()
            if self.force_reset_fov:
                intrinsic_color[0, 0] = intrinsic_color[0, 2] / math.tan(math.radians(90) / 2)
                intrinsic_color[1, 1] = intrinsic_color[1, 2] / math.tan(math.radians(90) / 2)
            cam2world = camera2world_list[sample_index]
            cam2scene.append(torch.from_numpy(self.world2scene @ cam2world).float())
            self.cam2scenes[sample_index] = cam2scene[-1]
            dims.append([img_h, img_w])
            intrinsics.append(torch.from_numpy(intrinsic_color).float())
            self.intrinsics[sample_index] = intrinsic_color
            self.intrinsics[sample_index] = torch.from_numpy(np.diag([self.image_dim[1] / img_w
                                                                         , self.image_dim[0] / img_h, 1]) @ self.intrinsics[sample_index]).float()
        self.scene2normscene = compute_world2normscene(
            torch.Tensor(dims).float(),
            torch.stack(intrinsics).float(),
            torch.stack(cam2scene).float(),
            max_depth=self.max_depth,
            rescale_factor=1.0
        )

        self.normscene_scale = self.scene2normscene[0, 0]
        for sample_index in sample_indices:
            self.cam2normscene[sample_index] = self.scene2normscene @ self.cam2scenes[sample_index]

        if self.split == "train" and not self.do_not_load:
            for sample_index in tqdm(self.train_indices, desc='dataload'):
                image, rays, semantics, instances, depth, _, probabilities, confidences, feat, room_mask = self.load_sample(sample_index)
                self.all_rgbs.append(image)
                self.all_rays.append(rays)
                self.all_semantics.append(semantics)
                self.all_probabilities.append(probabilities)
                self.all_confidences.append(confidences)
                self.all_instances.append(instances)
                self.all_masks.append(room_mask)
                if self.load_feat:
                    self.all_feats.append(feat)
                if self.load_depth:
                    self.all_depths.append(depth)
                self.all_origins.append(torch.ones_like(semantics) * sample_index)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_masks = torch.cat(self.all_masks, 0)
            self.all_semantics = torch.cat(self.all_semantics, 0)
            self.all_instances = torch.cat(self.all_instances, 0)
            self.all_probabilities = torch.cat(self.all_probabilities, 0)
            self.all_confidences = torch.cat(self.all_confidences, 0)
            if self.load_feat:
                self.all_feats = torch.cat(self.all_feats, 0)
            if self.load_depth:
                self.all_depths = torch.cat(self.all_depths, 0)
            self.all_origins = torch.cat(self.all_origins, 0)

        # for compatibility with PanopLi dataset
        self.segmentation_data = EasyDict({
            'fg_classes': [1], # one foreground class
            'bg_classes': [0], # one background class
            # 'instance_to_semantics': seg_data[dataset_ref.instance_to_semantic_key],
            'num_semantic_classes': 2,
            'num_instances': 1
        })
        
    def load_sample(self, sample_index):
        cam2normscene = self.cam2normscene[sample_index]
        image = Image.open(self.root_dir / "color" / f"{self.all_frame_names[sample_index]}.png")
        # noinspection PyTypeChecker
        image = torch.from_numpy(np.array(image.resize(self.image_dim[::-1], Image.LANCZOS)) / 255).float()
        if image.shape[-1] == 4:
            image = image[..., :3]
        
        semantics = np.load(self.root_dir / self.semantics_directory / f"{self.all_frame_names[sample_index]}.npy")
        instances = np.load(self.root_dir / self.instance_directory / f"{self.all_frame_names[sample_index]}.npy")
        # NOTE: the if-else below is to handle using "NON-groundtruth" or "groundtruth" labels for training
        if self.semantics_directory.split('_')[0] != 'semantic': # i.e. NON-groundtruth used
            confidences = np.load(self.root_dir / f"{self.semantics_directory.split('_')[0]}_probabilities" / f"{self.all_frame_names[sample_index]}.npy").astype(np.float32)
            ### HACK: for background pixels, set confidence to 1
            confidences[semantics == 0] = 1.0
        else: # using GT labels, so confidences are all ones.
            confidences = np.ones_like(semantics).astype(np.float32)
        
        # convert np array to PIL image
        semantics = Image.fromarray(semantics.astype(np.uint8))
        instances = Image.fromarray(instances.astype(np.int16))
        
        # resize semantics to image_dim
        semantics = torch.from_numpy(np.array(semantics.resize(self.image_dim[::-1], Image.NEAREST))).long()
        
        # resize confidences to image_dim
        confidences = torch.from_numpy(confidences).float() # h x w
        confidences = torch.nn.functional.interpolate(confidences.unsqueeze(0).unsqueeze(0), size=self.image_dim[::-1], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)        
        
        # probabilities is just one-hot encoding of semantics
        probabilities = torch.nn.functional.one_hot(semantics.long(), num_classes=self.num_semantics).float() 
        
        # dummy, for compatibility
        feat = torch.zeros(1)

        instances = torch.from_numpy(np.array(instances.resize(self.image_dim[::-1], Image.NEAREST))).long()
        # noinspection PyTypeChecker
        depth = torch.zeros(1)
        depth_cam = torch.zeros(1)
        if self.load_depth:
            raw_depth = np.array(Image.open(self.root_dir / "depth" / f"{self.all_frame_names[sample_index]}.png"))
            raw_depth = raw_depth.astype(np.float32) / 1000
            raw_depth[raw_depth > (self.max_depth / self.normscene_scale.item())] = (self.max_depth / self.normscene_scale.item())
            # noinspection PyTypeChecker
            depth_cam = torch.from_numpy(np.array(Image.fromarray(raw_depth).resize(self.image_dim[::-1], Image.NEAREST)))
            depth_cam_s = self.normscene_scale * depth_cam
            depth = depth_cam_s.float()

        directions = get_ray_directions_with_intrinsics(self.image_dim[0], self.image_dim[1], self.intrinsics[sample_index].numpy())
        rays_o, rays_d = get_rays(directions, cam2normscene)

        sphere_intersection_displacement = rays_intersect_sphere(rays_o, rays_d, r=1)  # fg is in unit sphere

        rays = torch.cat(
            [rays_o, rays_d, 0.01 *
             torch.ones_like(rays_o[:, :1]), sphere_intersection_displacement[:, None], ], 1,
        )
        room_mask_path = self.root_dir / "invalid" / f"{self.all_frame_names[sample_index]}.jpg"
        if room_mask_path.exists():
            room_mask = ~torch.from_numpy(np.array(Image.open(room_mask_path).resize(self.image_dim[::-1], Image.NEAREST)) > 0).bool()
        else:
            room_mask = torch.ones(rays.shape[0]).bool()
        return image.reshape(-1, 3), rays, semantics.reshape(-1), instances.reshape(-1), depth.reshape(-1), \
               depth_cam.reshape(-1), probabilities.reshape(-1, probabilities.shape[-1]), confidences.reshape(-1),\
               feat.reshape(-1, feat.shape[-1]), room_mask.reshape(-1)

    def export_point_cloud(self, output_path, subsample=1, export_semantics=False, export_bbox=False):
        super().export_point_cloud(output_path, subsample, export_semantics, export_bbox)

    @property
    def things_filtered(self):
        return set([i for i in range(len(self.is_thing)) if self.is_thing[i]]) - set(self.faulty_classes)

    @property
    def stuff_filtered(self):
        return set([i for i in range(len(self.is_thing)) if not self.is_thing[i]]) - set(self.faulty_classes)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if self.split == 'val' or self.split == 'test':
            sample_idx = self.val_indices[idx % len(self.val_indices)]
            semantics = np.load(self.root_dir / "semantic" / f"{self.all_frame_names[sample_idx]}.npy")
            instances = np.load(self.root_dir / "instance" / f"{self.all_frame_names[sample_idx]}.npy")
            semantics = Image.fromarray(semantics.astype(np.uint8))
            instances = Image.fromarray(instances.astype(np.int16))
            semantics = torch.from_numpy(np.array(semantics.resize(self.image_dim[::-1], Image.NEAREST))).long().reshape(-1)
            instances = torch.from_numpy(np.array(instances.resize(self.image_dim[::-1], Image.NEAREST))).long().reshape(-1)
            sample['rs_semantics'] = semantics
            sample['rs_instances'] = instances
        return sample


class InconsistentMOSSingleDataset(MOSDataset):

    def __init__(self, root_dir, split, image_dim, max_depth, overfit=False, num_val_samples=8, max_rays=512, semantics_dir='filtered_semantics',
                 instance_dir='filtered_instance_inc', instance_to_semantic_key='instance_to_semantic_inc', create_seg_data_func=create_segmentation_data_base, subsample_frames=1):
        super().__init__(root_dir, split, image_dim, max_depth, overfit, num_val_samples, semantics_dir=semantics_dir,
                         instance_dir=instance_dir, instance_to_semantic_key=instance_to_semantic_key, create_seg_data_func=create_seg_data_func, subsample_frames=subsample_frames)
        print('Preparing InconsistentPanopLiDataset...')
        all_rays_view = self.all_rays.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1], -1)
        all_instances_view = self.all_instances.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        all_confidences_view = self.all_confidences.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        all_masks_view = self.all_masks.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        all_confidences_view[~all_masks_view] = 0
        all_rays = []
        all_instances = []
        all_confidences = []
        for i in range(len(self.train_indices)):
            mask = all_instances_view[i] != 0
            if mask.sum() > 0:
                all_rays.append(all_rays_view[i][mask, :])
                all_instances.append(all_instances_view[i][mask])
                all_confidences.append(all_confidences_view[i][mask])
        self.all_rays = all_rays
        self.all_instances = all_instances
        self.all_confidences = all_confidences
        self.max_rays = max_rays

    def __getitem__(self, idx):
        selected_rays = self.all_rays[idx]
        selected_instances = self.all_instances[idx]
        selected_confidences = self.all_confidences[idx]
        if selected_rays.shape[0] > self.max_rays:
            sampled_indices = random.sample(range(selected_rays.shape[0]), self.max_rays)
            selected_rays = selected_rays[sampled_indices, :]
            selected_instances = selected_instances[sampled_indices]
            selected_confidences = selected_confidences[sampled_indices]
        sample = {
            f"rays": selected_rays,
            f"instances": selected_instances,
            f"confidences": selected_confidences,
        }
        return sample

    def __len__(self):
        return len(self.all_rays)

    @staticmethod
    def collate_fn(batch):
        return {
            "rays": [x["rays"] for x in batch],
            "instances": [x["instances"] for x in batch],
            "confidences": [x["confidences"] for x in batch],
        }


def create_segmentation_data_panopli(dataset_ref, seg_data):
    seg_data_dict = EasyDict({
        'fg_classes': sorted(seg_data['fg_classes']),
        'bg_classes': sorted(seg_data['bg_classes']),
        'instance_to_semantics': seg_data[dataset_ref.instance_to_semantic_key],
        'num_semantic_classes': len(seg_data['fg_classes'] + seg_data['bg_classes']),
        'num_instances': len(seg_data['fg_classes'])
    })
    return seg_data_dict


def create_segmentation_data_panopli_with_valid(dataset_ref, seg_data):
    seg_data_dict = create_segmentation_data_panopli(dataset_ref, seg_data)
    seg_data_dict['instance_is_valid'] = seg_data['m2f_sem_valid_instance']
    return seg_data_dict


def create_segmentation_data_panopli_mmdet(dataset_ref, seg_data):
    dataset_ref.bounding_boxes = process_bounding_box_dict(seg_data['mmdet_bboxes'], dataset_ref.scene2normscene.numpy())
    return EasyDict({
        'fg_classes': sorted(seg_data['fg_classes']),
        'bg_classes': sorted(seg_data['bg_classes']),
        'instance_to_semantics': {
            (i + 1): seg_data['mmdet_bboxes'][i]['class']
            for i in range(dataset_ref.bounding_boxes.ids.shape[0])
        },
        'num_semantic_classes': len(seg_data['fg_classes'] + seg_data['bg_classes']),
        'num_instances': dataset_ref.bounding_boxes.ids.shape[0]
    })


def create_segmentation_data_panopli_gt(dataset_ref, seg_data):
    dataset_ref.bounding_boxes = process_bounding_box_dict(seg_data['gt_bboxes'], dataset_ref.scene2normscene.numpy())
    return EasyDict({
        'fg_classes': sorted(seg_data['fg_classes']),
        'bg_classes': sorted(seg_data['bg_classes']),
        'instance_to_semantics': {
            (i + 1): seg_data['gt_bboxes'][i]['class']
            for i in range(dataset_ref.bounding_boxes.ids.shape[0])
        },
        'num_semantic_classes': len(seg_data['fg_classes'] + seg_data['bg_classes']),
        'num_instances': dataset_ref.bounding_boxes.ids.shape[0]
    })


class SegmentMOSDataset(MOSDataset):

    def __init__(self, root_dir, split, image_dim, max_depth, overfit=False, num_val_samples=8, max_rays=512, semantics_dir='detic_semantic',
                 instance_dir='detic_instance', instance_to_semantic_key='instance_to_semantic_inc', create_seg_data_func=create_segmentation_data_base, subsample_frames=1):
        super().__init__(root_dir, split, image_dim, max_depth, overfit, num_val_samples, semantics_dir=semantics_dir,
                         instance_dir=instance_dir, instance_to_semantic_key=instance_to_semantic_key, create_seg_data_func=create_seg_data_func, subsample_frames=subsample_frames)
        print('Preparing SegmentPanopLi...')
        all_rays_view = self.all_rays.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1], -1)
        all_confidences_view = self.all_confidences.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        all_masks_view = self.all_masks.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        all_confidences_view[~all_masks_view] = 0
        all_rays = []
        all_confidences = []
        all_ones = []
        for i in range(len(self.train_indices)):
            segments = np.load(self.root_dir / "detic_instance" / f"{self.all_frame_names[self.train_indices[i]]}.npy")
            segments = Image.fromarray(segments.astype(np.int16))
            segments = torch.from_numpy(np.array(segments.resize(self.image_dim[::-1], Image.NEAREST))).long().reshape(-1)
            for s in torch.unique(segments):
                if s.item() != 0:
                    all_rays.append(all_rays_view[i][segments == s, :])
                    all_confidences.append(all_confidences_view[i][segments == s])
                    all_ones.append(torch.ones(all_confidences[-1].shape[0]).long())
        self.all_rays = all_rays
        self.all_confidences = all_confidences
        self.all_ones = all_ones
        self.max_rays = max_rays
        self.enabled = False

    def __getitem__(self, idx):
        if self.enabled:
            selected_rays = self.all_rays[idx]
            selected_confidences = self.all_confidences[idx]
            selected_ones = self.all_ones[idx]
            if selected_rays.shape[0] > self.max_rays:
                sampled_indices = random.sample(range(selected_rays.shape[0]), self.max_rays)
                selected_rays = selected_rays[sampled_indices, :]
                selected_confidences = selected_confidences[sampled_indices]
                selected_ones = selected_ones[sampled_indices]
            sample = {
                f"rays": selected_rays,
                f"confidences": selected_confidences,
                f"group": selected_ones,
            }
        else:
            sample = {
                f"rays": [0],
                f"confidences": [0],
                f"group": [0],
            }
        return sample

    def __len__(self):
        return len(self.all_rays)

    @staticmethod
    def collate_fn(batch):
        return {
            "rays": [x["rays"] for x in batch],
            "confidences": [x["confidences"] for x in batch],
            "group": [batch[i]['group'] * i for i in range(len(batch))]
        }
