# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import sys
import time
from pathlib import Path
import pickle
from typing import Any
import torch
import omegaconf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
from sklearn.cluster import MeanShift
from scipy.stats import gaussian_kde
from hdbscan import HDBSCAN

sys.path.append(".")
from dataset import PanopLiDataset, create_segmentation_data_panopli
from dataset.many_object_scenes import MOSDataset
from model.radiance_field.tensoRF import TensorVMSplit, MLPRenderInstanceFeature
from model.renderer.panopli_tensoRF_renderer import TensoRFRenderer
from trainer import visualize_panoptic_outputs
from util.camera import distance_to_depth
from util.misc import get_parameters_from_state_dict
from util.points_util import savePlyFromPtsRGB


def render_panopli_checkpoint(
        config: Any,
        trajectory_name: Any,
        test_only: bool = False,
        bandwidth: float = 0.15,
        use_dbscan: bool = False,
        segmentwise: bool = False,
        subpath: Any = None,
        use_silverman: bool = False,
        cluster_size: int = 500,
        cached_centroids_path: Any = None,
):
    output_dir = output_dirname(config, trajectory_name, test_only, use_dbscan, segmentwise)
    if subpath is not None:
        output_dir = output_dir / subpath
    print(output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    if config.dataset_class == "panopli":
        test_set = PanopLiDataset(
            Path(config.dataset_root), "test", (config.image_dim[0], config.image_dim[1]),
            config.max_depth, overfit=config.overfit, semantics_dir='m2f_semantics',
            instance_dir='m2f_instance', instance_to_semantic_key='m2f_instance_to_semantic',
            create_seg_data_func=create_segmentation_data_panopli,
            subsample_frames=config.subsample_frames
        )
    elif config.dataset_class == "mos":
        test_set = MOSDataset(
            Path(config.dataset_root), "test", (config.image_dim[0], config.image_dim[1]),
            config.max_depth, overfit=config.overfit, semantics_dir='detic_semantic',
            instance_dir='detic_instance', instance_to_semantic_key=None,
            create_seg_data_func=None, subsample_frames=config.subsample_frames
        )

    H, W, alpha = config.image_dim[0], config.image_dim[1], 0.65
    # whether to render the test set or a predefined trajectory through the scene
    if test_only:
        trajectory_set = test_set
    else:
        trajectory_set = test_set.get_trajectory_set(trajectory_name, True)
    trajectory_loader = DataLoader(trajectory_set, shuffle=False, num_workers=0, batch_size=1)
    checkpoint = torch.load(config.resume, map_location="cpu")
    state_dict = checkpoint['state_dict']
    total_classes = len(test_set.segmentation_data.bg_classes) + len(test_set.segmentation_data.fg_classes)
    output_mlp_semantics = torch.nn.Identity() if config.semantic_weight_mode != "softmax" else torch.nn.Softmax(dim=-1)
    model = TensorVMSplit(
        [config.min_grid_dim, config.min_grid_dim, config.min_grid_dim],
        num_semantics_comps=(32, 32, 32), num_instance_comps=(32, 32, 32), num_semantic_classes=total_classes,
        dim_feature_instance=2*config.max_instances if config.instance_loss_mode=="slow_fast" else config.max_instances,
        output_mlp_semantics=output_mlp_semantics, use_semantic_mlp=config.use_mlp_for_semantics,  
        use_instance_mlp=config.use_mlp_for_instances,
        use_distilled_features_semantic=config.use_distilled_features_semantic, use_distilled_features_instance=config.use_distilled_features_instance,
        pe_sem=config.pe_sem, pe_ins=config.pe_ins,
        slow_fast_mode=config.instance_loss_mode=="slow_fast", use_proj=config.use_proj,
    )
    renderer = TensoRFRenderer(
        test_set.scene_bounds, [config.min_grid_dim, config.min_grid_dim, config.min_grid_dim],
        semantic_weight_mode=config.semantic_weight_mode
    )
    renderer.load_state_dict(get_parameters_from_state_dict(state_dict, "renderer"))
    for epoch in config.grid_upscale_epochs[::-1]:
        if checkpoint['epoch'] >= epoch:
            model.upsample_volume_grid(renderer.grid_dim)
            renderer.update_step_size(renderer.grid_dim)
            break

    model.load_state_dict(get_parameters_from_state_dict(state_dict, "model"))

    model = model.to(device)
    renderer = renderer.to(device)

    # disable this for fast rendering (just add more steps along the ray)
    renderer.update_step_ratio(renderer.step_ratio * 0.5)

    all_points_rgb, all_points_semantics, all_points_instances, all_points_depth = [], [], [], []
    all_instance_features, all_thing_features, all_slow_features = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(trajectory_loader)):
            batch['rays'] = batch['rays'].squeeze(0).to(device)
            concated_outputs = []
            outputs = []
            # infer semantics and surrogate ids
            for i in range(0, batch['rays'].shape[0], config.chunk):
                out_rgb_, out_semantics_, out_instances_, out_depth_, _, _ = renderer(
                    model, batch['rays'][i: i + config.chunk], config.perturb, test_set.white_bg, False)
                outputs.append([out_rgb_, out_semantics_, out_instances_, out_depth_])
            for i in range(len(outputs[0])):
                concated_outputs.append(torch.cat([outputs[j][i] for j in range(len(outputs))], dim=0))
            p_rgb, p_semantics, p_instances, p_dist = concated_outputs
            p_depth = distance_to_depth(test_set.intrinsics[0], p_dist.view(H, W))

            if config.use_delta:
                points_xyz = batch['rays'][...,0:3] + p_dist[...,None] * batch['rays'][...,3:6] # B x 3
                p_instances = p_instances + points_xyz

            if model.slow_fast_mode:
                slow_features = p_instances[...,config.max_instances:] 
                all_slow_features.append(slow_features)
                p_instances = p_instances[...,0:config.max_instances] # keep fast features only

            all_instance_features.append(p_instances)

            all_points_rgb.append(p_rgb)
            all_points_semantics.append(p_semantics)
            all_points_depth.append(p_depth)

            # create surrogate ids
            p_instances = create_instances_from_semantics(p_instances, p_semantics, test_set.segmentation_data.fg_classes)
            all_thing_features.append(p_instances)

    all_instance_features = torch.cat(all_instance_features, dim=0).cpu().numpy()
    np.save(output_dir / "instance_features.npy", all_instance_features)

    all_thing_features = torch.cat(all_thing_features, dim=0).cpu().numpy() # N x d
    np.save(output_dir / "thing_features.npy", all_thing_features)

    if model.slow_fast_mode:
        all_slow_features = torch.cat(all_slow_features, dim=0).cpu().numpy()
        np.save(output_dir / "slow_features.npy", all_slow_features)

    use_cached_centroids = cached_centroids_path is not None
    if use_cached_centroids:
        with open(cached_centroids_path, 'rb') as f:
            all_centroids = pickle.load(f)
        all_points_instances = assign_clusters(
            all_thing_features, all_points_semantics, all_centroids, device, num_images=len(all_points_rgb))
    else:
        ### Perform clustering on the thing features
        if not segmentwise:
            all_points_instances = cluster(
                all_thing_features, bandwidth, device, num_images=len(all_points_rgb),
                use_dbscan=use_dbscan, use_silverman=use_silverman, cluster_size=cluster_size
            )
        else:
            all_points_instances, _ = cluster_segmentwise(
                all_thing_features, all_points_semantics, bandwidth, device, num_images=len(all_points_rgb),
                use_dbscan=use_dbscan, use_silverman=use_silverman, cluster_size=cluster_size
            )

    # save outputs
    (output_dir / "vis_semantics_and_surrogate").mkdir(exist_ok=True)
    (output_dir / "pred_semantics").mkdir(exist_ok=True)
    (output_dir / "pred_surrogateid").mkdir(exist_ok=True)
    for i, _ in enumerate(all_points_rgb):
        name = f"{test_set.all_frame_names[test_set.val_indices[i]]}.png" if test_only else f"{i:04d}.png"
        p_rgb, p_semantics, p_instances, p_depth = all_points_rgb[i], all_points_semantics[i], all_points_instances[i], all_points_depth[i]
        stack = visualize_panoptic_outputs(
            p_rgb, p_semantics, p_instances, p_depth, None, None, None,
            H, W, thing_classes=test_set.segmentation_data.fg_classes, visualize_entropy=False
        )
        output_semantics_with_invalid = p_semantics.detach().argmax(dim=1)
        grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=5).permute((1, 2, 0)).contiguous()
        grid = (grid * 255).cpu().numpy().astype(np.uint8)

        Image.fromarray(grid).save(output_dir / "vis_semantics_and_surrogate" / name)
        Image.fromarray(output_semantics_with_invalid.reshape(H, W).cpu().numpy().astype(np.uint8)).save(output_dir / "pred_semantics" / name)
        Image.fromarray(p_instances.argmax(dim=1).reshape(H, W).cpu().numpy().astype(np.uint16)).save(output_dir / "pred_surrogateid" / name)


def output_dirname(config, trajectory_name, test_only, use_dbscan, segmentwise):
    output_dir = Path("runs") / f"{Path(config.dataset_root).stem}_{trajectory_name if not test_only else 'test'}_{Path(config.experiment)}{'_dbscan' if use_dbscan else ''}{'_seg' if segmentwise else ''}"
    return output_dir


def cluster(all_thing_features, bandwidth, device, num_images=None, use_dbscan=False,
            use_silverman=False, cluster_size=500):
    thing_mask = all_thing_features[...,0] == -float('inf')
    features = all_thing_features[thing_mask]
    features = features[:,1:]
    all_thing_features = all_thing_features[:,1:]
    
    # remove outliers assuming Gaussian distribution
    centmean, centstd = features.mean(axis=0), features.std(axis=0)
    outlier_mask = np.all(np.abs(features - centmean) < 3 * centstd, axis=1)
    centers_filtered = features[outlier_mask]
    print("Num centers pre-filtering: ", features.shape[0], features.min(axis=0), features.max(axis=0))
    print("Num centers post-filtering: ", centers_filtered.shape[0], centers_filtered.min(axis=0), centers_filtered.max(axis=0))
    rescaling_bias = centers_filtered.min(axis=0)
    rescaling_factor = 1/(centers_filtered.max(axis=0) - centers_filtered.min(axis=0))
    centers_rescaled = (centers_filtered - rescaling_bias) * rescaling_factor
    # perform clustering
    num_points = 50000
    fps_points_indices = np.random.choice(centers_rescaled.shape[0], num_points, replace=False)
    fps_points_rescaled = centers_rescaled[fps_points_indices]
    
    if not use_dbscan:
        t1_ms = time.time()
        if use_silverman:
            kde = gaussian_kde(fps_points_rescaled.T, bw_method='silverman')
            bandwidth_ = kde.covariance_factor()
            print("Using Silverman bandwidth: ", bandwidth_)
        else:
            bandwidth_ = bandwidth
        clustering = MeanShift(bandwidth=bandwidth_, cluster_all=False, bin_seeding=True,
                               min_bin_freq=10).fit(fps_points_rescaled)
        t2_ms = time.time()
        print(f"MeanShift took {t2_ms-t1_ms} seconds")
        labels = clustering.labels_
        centroids = clustering.cluster_centers_
        all_labels = clustering.predict(
            (all_thing_features.reshape(-1, all_thing_features.shape[-1]) - rescaling_bias) * rescaling_factor
        )
    else: # Use HDBSCAN
        t1_dbscan = time.time()
        clusterer = HDBSCAN(min_cluster_size=cluster_size, min_samples=1, prediction_data=True,
                                    allow_single_cluster=True).fit(fps_points_rescaled)
        t2_dbscan = time.time()
        print(f"HDBSCAN took {t2_dbscan-t1_dbscan} seconds")
        labels = clusterer.labels_
        centroids = np.stack([clusterer.weighted_cluster_centroid(cluster_id=cluster_id) \
                              for cluster_id in np.unique(labels) if cluster_id != -1])
        distances = torch.zeros((all_thing_features.shape[0], centroids.shape[0]), device=device)
        chunksize = 10**7
        all_thing_features_rescaled = (all_thing_features.reshape(-1, all_thing_features.shape[-1]) - rescaling_bias) * rescaling_factor
        for i in range(0, all_thing_features.shape[0], chunksize):
            distances[i:i+chunksize] = torch.cdist(
                torch.FloatTensor(all_thing_features_rescaled[i:i+chunksize]).to(device),
                torch.FloatTensor(centroids).to(device)
            )
        all_labels = torch.argmin(distances, dim=-1).cpu().numpy()

    all_labels[~thing_mask] = -1
    # to one hot
    all_labels = all_labels + 1 # -1,0,...,K-1 -> 0,1,...,K
    all_labels_onehot = np.zeros((all_labels.shape[0], centroids.shape[0]+1))
    all_labels_onehot[np.arange(all_labels.shape[0]), all_labels] = 1
    all_points_instances = torch.from_numpy(all_labels_onehot).view(num_images, -1, centroids.shape[0]+1).to(device)
    return all_points_instances

def cluster_segmentwise(all_thing_features, all_points_semantics, bandwidth, device, num_images=None,
                        use_dbscan=False, use_silverman=False, cluster_size=500):
    all_points_semantics = torch.cat(all_points_semantics, dim=0).argmax(dim=-1).cpu().numpy()

    thing_mask = all_thing_features[...,0] == -float('inf')
    features = all_thing_features[thing_mask]
    features = features[:,1:]
    all_thing_features = all_thing_features[:,1:]

    thing_semantics = all_points_semantics[thing_mask]
    thing_classes = np.unique(thing_semantics)

    all_labels = np.zeros(all_thing_features.shape[0], dtype=np.int32)
    all_thing_labels = np.zeros(features.shape[0], dtype=np.int32)
    max_label = 0
    all_centroids = []
    for thing_cls in thing_classes:
        thing_cls_mask = thing_semantics == thing_cls
        thing_cls_features = features[thing_cls_mask] # features of this thing class

        # remove outliers assuming Gaussian distribution
        centmean, centstd = thing_cls_features.mean(axis=0), thing_cls_features.std(axis=0)
        outlier_mask = np.all(np.abs(thing_cls_features - centmean) < 3 * centstd, axis=1)
        centers_filtered = thing_cls_features[outlier_mask]
        # if centers_filtered is empty
        if centers_filtered.shape[0] == 0:
            thing_cls_all_labels = -1 * np.ones(thing_cls_features.shape[0], dtype=np.int32)
            all_thing_labels[thing_cls_mask] = thing_cls_all_labels
            continue

        print(f"[THING CLS: {thing_cls}] Num centers pre-filtering: ", thing_cls_features.shape[0], features.min(axis=0), features.max(axis=0))
        print(f"[THING CLS: {thing_cls}] Num centers post-filtering: ", centers_filtered.shape[0], centers_filtered.min(axis=0), centers_filtered.max(axis=0))
        rescaling_bias = centers_filtered.min(axis=0)
        rescaling_factor = 1/(centers_filtered.max(axis=0) - centers_filtered.min(axis=0))
        centers_rescaled = (centers_filtered - rescaling_bias) * rescaling_factor
        # perform clustering
        num_points = 50000
        if centers_rescaled.shape[0] < num_points:
            fps_points_indices = np.arange(centers_rescaled.shape[0])
        else:
            fps_points_indices = np.random.choice(centers_rescaled.shape[0], num_points, replace=False)
        fps_points_rescaled = centers_rescaled[fps_points_indices]
        
        if not use_dbscan:
            if fps_points_rescaled.shape[0] < 100: # too few points for MeanShift
                thing_cls_all_labels = -1 * np.ones(thing_cls_features.shape[0], dtype=np.int32)
            else:
                t1_ms = time.time()
                if use_silverman:
                    kde = gaussian_kde(fps_points_rescaled.T, bw_method='silverman')
                    bandwidth_ = kde.covariance_factor()
                    print("Using Silverman bandwidth: ", bandwidth_)
                else:
                    bandwidth_ = bandwidth
                clustering = MeanShift(bandwidth=bandwidth_, cluster_all=False, bin_seeding=True, 
                                       min_bin_freq=10).fit(fps_points_rescaled)
                t2_ms = time.time()
                print(f"MeanShift took {t2_ms-t1_ms} seconds")
                labels = clustering.labels_
                centroids = clustering.cluster_centers_
                thing_cls_all_labels = clustering.predict((thing_cls_features.reshape(-1, thing_cls_features.shape[-1]) - rescaling_bias) * rescaling_factor)
        else: # Use HDBSCAN
            t1_dbscan = time.time()
            clusterer = HDBSCAN(min_cluster_size=cluster_size, min_samples=1, prediction_data=True,
                                allow_single_cluster=True).fit(fps_points_rescaled)
            t2_dbscan = time.time()
            print(f"HDBSCAN took {t2_dbscan-t1_dbscan} seconds")
            labels = clusterer.labels_
            if np.any(labels != -1): # i.e. if there are clusters
                centroids = np.stack([clusterer.weighted_cluster_centroid(cluster_id=cluster_id) \
                                      for cluster_id in np.unique(labels) if cluster_id != -1])
                distances = torch.zeros((thing_cls_features.shape[0], centroids.shape[0]), device=device)
                chunksize = 10**7
                thing_cls_features_rescaled = (thing_cls_features.reshape(-1, thing_cls_features.shape[-1]) - rescaling_bias) * rescaling_factor
                for i in range(0, thing_cls_features.shape[0], chunksize):
                    distances[i:i+chunksize] = torch.cdist(
                        torch.FloatTensor(thing_cls_features_rescaled[i:i+chunksize]).to(device),
                        torch.FloatTensor(centroids).to(device)
                    )
                thing_cls_all_labels = torch.argmin(distances, dim=-1).cpu().numpy()
            else:
                thing_cls_all_labels = -1 * np.ones(thing_cls_features.shape[0], dtype=np.int32)

        # rescale back the centroids
        centroids_scaled_back = centroids / rescaling_factor + rescaling_bias
        all_centroids.append(centroids_scaled_back)
        
        # assign labels
        # if thing_cls_all_labels=-1, keep it as -1
        # else add max_label and assign it to thing_cls_all_labels
        thing_cls_all_labels[thing_cls_all_labels != -1] += max_label
        if np.any(thing_cls_all_labels != -1): # i.e. if there are clusters
            max_label = thing_cls_all_labels.max() + 1
        all_thing_labels[thing_cls_mask] = thing_cls_all_labels

    all_labels[thing_mask] = all_thing_labels
    all_labels[~thing_mask] = -1 # assign -1 to stuff points
    all_labels = all_labels + 1 # -1,0,...,K-1 -> 0,1,...,K
    # num_unique_labels = np.unique(all_labels).shape[0] 
    # NOTE: the above line has a problem when there is no stuff class (i.e. all_labels > 0)
    num_unique_labels = all_labels.max() + 1 # 0,1,...,K
    print("Num unique labels: ", num_unique_labels)
    all_labels_onehot = np.zeros((all_labels.shape[0], num_unique_labels))
    all_labels_onehot[np.arange(all_labels.shape[0]), all_labels] = 1
    all_points_instances = torch.from_numpy(all_labels_onehot).view(num_images, -1, num_unique_labels).to(device)

    all_centroids = np.concatenate(all_centroids, axis=0) # (num_clusters, 3)
    return all_points_instances, all_centroids


def assign_clusters(all_thing_features, all_points_semantics, all_centroids, device, num_images=None):
    all_points_semantics = torch.cat(all_points_semantics, dim=0).argmax(dim=-1).cpu().numpy()

    thing_mask = all_thing_features[...,0] == -float('inf')
    features = all_thing_features[thing_mask]
    features = features[:,1:]
    all_thing_features = all_thing_features[:,1:]

    thing_semantics = all_points_semantics[thing_mask]
    thing_classes = np.unique(thing_semantics)

    all_labels = np.zeros(all_thing_features.shape[0], dtype=np.int32)
    all_thing_labels = np.zeros(features.shape[0], dtype=np.int32)
    max_label = 0
    for thing_cls in thing_classes:
        thing_cls_mask = thing_semantics == thing_cls
        thing_cls_features = features[thing_cls_mask] # features of this thing class

        centroids = all_centroids[thing_cls]
        distances = torch.zeros((thing_cls_features.shape[0], centroids.shape[0]), device=device)
        chunksize = 10**7
        thing_cls_features_reshaped = thing_cls_features.reshape(-1, thing_cls_features.shape[-1])
        for i in range(0, thing_cls_features.shape[0], chunksize):
            distances[i:i+chunksize] = torch.cdist(
                torch.FloatTensor(thing_cls_features_reshaped[i:i+chunksize]).to(device),
                torch.FloatTensor(centroids).to(device)
            )
        thing_cls_all_labels = torch.argmin(distances, dim=-1).cpu().numpy()
        
        # assign labels
        # if thing_cls_all_labels=-1, keep it as -1
        # else add max_label and assign it to thing_cls_all_labels
        thing_cls_all_labels[thing_cls_all_labels != -1] += max_label
        if np.any(thing_cls_all_labels != -1): # i.e. if there are clusters
            max_label = thing_cls_all_labels.max() + 1
        all_thing_labels[thing_cls_mask] = thing_cls_all_labels

    all_labels[thing_mask] = all_thing_labels
    all_labels[~thing_mask] = -1 # assign -1 to stuff points
    all_labels = all_labels + 1 # -1,0,...,K-1 -> 0,1,...,K
    # num_unique_labels = np.unique(all_labels).shape[0] 
    # NOTE: the above line has a problem when there is no stuff class (i.e. all_labels > 0)
    num_unique_labels = all_labels.max() + 1 # 0,1,...,K
    print("Num unique labels: ", num_unique_labels)
    all_labels_onehot = np.zeros((all_labels.shape[0], num_unique_labels))
    all_labels_onehot[np.arange(all_labels.shape[0]), all_labels] = 1
    all_points_instances = torch.from_numpy(all_labels_onehot).view(num_images, -1, num_unique_labels).to(device)

    return all_points_instances


def create_instances_from_semantics(instances, semantics, thing_classes):
    stuff_mask = ~torch.isin(semantics.argmax(dim=1), torch.tensor(thing_classes).to(semantics.device))
    padded_instances = torch.ones((instances.shape[0], instances.shape[1] + 1), device=instances.device) * -float('inf')
    padded_instances[:, 1:] = instances
    padded_instances[stuff_mask, 0] = float('inf')
    return padded_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--render_trajectory", action="store_true")
    parser.add_argument("--bandwidth", type=float, default=0.15, required=False)
    parser.add_argument("--cluster_size", type=int, default=500, required=False,
                        help="min_cluster_size for HDBSCAN")
    parser.add_argument("--use_dbscan", action="store_true")
    parser.add_argument("--segmentwise", action="store_true")
    parser.add_argument("--subsample", type=int, default=1, required=False)
    parser.add_argument("--use_silverman", action="store_true")
    parser.add_argument("--cached_centroids_path", type=str, required=False)
    args = parser.parse_args()

    # needs a predefined trajectory named trajectory_blender in case test_only = False
    cfg = omegaconf.OmegaConf.load(Path(args.ckpt_path).parents[1] / "config.yaml")
    cfg.resume = args.ckpt_path
    TEST_MODE = not args.render_trajectory
    cfg.subsample_frames = args.subsample

    cfg.image_dim = [256, 384]    
    if isinstance(cfg.image_dim, int):
        cfg.image_dim = [cfg.image_dim, cfg.image_dim]

    render_panopli_checkpoint(
        cfg, "trajectory_blender", test_only=TEST_MODE, bandwidth=args.bandwidth,
        use_dbscan=args.use_dbscan, segmentwise=args.segmentwise,
        use_silverman=args.use_silverman, cluster_size=args.cluster_size,
        cached_centroids_path=args.cached_centroids_path)
