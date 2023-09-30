# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from pathlib import Path
from dataset.base import BaseDataset, create_segmentation_data_sem, InconsistentBaseDataset, InconsistentSingleBaseDataset
from dataset.panopli import PanopLiDataset, InconsistentPanopLiDataset, InconsistentPanopLiSingleDataset, create_segmentation_data_panopli, SegmentPanopLiDataset
from dataset.many_object_scenes import MOSDataset, InconsistentMOSSingleDataset, SegmentMOSDataset


def get_dataset(config, load_only_val=False, use_gt_inssem=False):
    if config.dataset_class == "panopli":
        if use_gt_inssem:
            instance_dir, semantics_dir, instance_to_semantic_key = 'rs_instance', 'rs_semantics', 'rs_instance_to_semantic'
        else:
            instance_dir, semantics_dir, instance_to_semantic_key = 'm2f_instance', 'm2f_semantics', 'm2f_instance_to_semantic'
        train_set = None
        if not load_only_val:
            train_set = PanopLiDataset(Path(config.dataset_root), "train", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir=semantics_dir,
                                       load_feat=config.use_distilled_features_semantic or config.use_distilled_features_instance, 
                                       feature_type=config.feature_type, # "nearest" (default) or "bilinear"
                                       instance_dir=instance_dir, instance_to_semantic_key=instance_to_semantic_key,
                                       create_seg_data_func=create_segmentation_data_panopli, subsample_frames=config.subsample_frames)
        val_set = PanopLiDataset(Path(config.dataset_root), "val", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir=semantics_dir,
                                 instance_dir=instance_dir, instance_to_semantic_key=instance_to_semantic_key, create_seg_data_func=create_segmentation_data_panopli,
                                 subsample_frames=config.subsample_frames)
        return train_set, val_set
    elif config.dataset_class == "mos": # Many-Object-Scenes
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = 'detic_instance', 'detic_semantic'
        train_set = None
        if not load_only_val:
            train_set = MOSDataset(Path(config.dataset_root), "train", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir=semantics_dir,
                                   load_feat=False, feature_type=None, # features not used for MOS
                                   instance_dir=instance_dir, instance_to_semantic_key=None,
                                   create_seg_data_func=None, subsample_frames=config.subsample_frames)
        val_set = MOSDataset(Path(config.dataset_root), "val", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir=semantics_dir,
                             instance_dir=instance_dir, instance_to_semantic_key=None, create_seg_data_func=None,
                             subsample_frames=config.subsample_frames)
        return train_set, val_set
    raise NotImplementedError


def get_inconsistent_single_dataset(config, use_gt_inssem=False):
    if config.dataset_class == "panopli":
        if use_gt_inssem:
            instance_dir, semantics_dir, instance_to_semantic_key = 'rs_instance', 'rs_semantics', 'rs_instance_to_semantic'
        else:
            instance_dir, semantics_dir, instance_to_semantic_key = 'm2f_instance', 'm2f_semantics', 'm2f_instance_to_semantic'
        return InconsistentPanopLiSingleDataset(Path(config.dataset_root), "train", (128, 128), config.max_depth, overfit=config.overfit,
                                                max_rays=config.max_rays_instances, semantics_dir=semantics_dir, instance_dir=instance_dir, instance_to_semantic_key=instance_to_semantic_key,
                                                create_seg_data_func=create_segmentation_data_panopli, subsample_frames=config.subsample_frames)
    elif config.dataset_class == "mos":
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = 'detic_instance', 'detic_semantic'
        return InconsistentMOSSingleDataset(Path(config.dataset_root), "train", (128, 128), config.max_depth, overfit=config.overfit,
                                            max_rays=config.max_rays_instances, semantics_dir=semantics_dir, instance_dir=instance_dir, instance_to_semantic_key=None,
                                            create_seg_data_func=None, subsample_frames=config.subsample_frames)
    raise NotImplementedError


def get_segment_dataset(config, use_gt_inssem=False):
    if config.dataset_class == "panopli":
        if use_gt_inssem:
            instance_dir, semantics_dir, instance_to_semantic_key = 'rs_instance', 'rs_semantics', 'rs_instance_to_semantic'
        else:
            instance_dir, semantics_dir, instance_to_semantic_key = 'm2f_instance', 'm2f_semantics', 'm2f_instance_to_semantic'
        return SegmentPanopLiDataset(Path(config.dataset_root), "train", (128, 128), config.max_depth, overfit=config.overfit,
                                     max_rays=config.max_rays_segments, semantics_dir=semantics_dir, instance_dir=instance_dir, instance_to_semantic_key=instance_to_semantic_key,
                                     create_seg_data_func=create_segmentation_data_panopli, subsample_frames=config.subsample_frames)
    elif config.dataset_class == "mos":
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = 'detic_instance', 'detic_semantic'
        return SegmentMOSDataset(Path(config.dataset_root), "train", (128, 128), config.max_depth, overfit=config.overfit,
                                 max_rays=config.max_rays_segments, semantics_dir=semantics_dir, instance_dir=instance_dir, instance_to_semantic_key=None,
                                 create_seg_data_func=None, subsample_frames=config.subsample_frames)
    raise NotImplementedError
