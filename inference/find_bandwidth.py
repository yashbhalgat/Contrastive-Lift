import argparse
import json
from pathlib import Path
import sys
import time

import hdbscan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
from PIL import Image
from scipy.stats import gaussian_kde
from sklearn.cluster import MeanShift
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

sys.path.append(".")
from dataset import PanopLiDataset, create_segmentation_data_panopli
from dataset.many_object_scenes import MOSDataset
from model.radiance_field.tensoRF import TensorVMSplit
from model.renderer.panopli_tensoRF_renderer import TensoRFRenderer
from trainer import visualize_panoptic_outputs
from util.camera import distance_to_depth
from util.misc import get_parameters_from_state_dict
from dataset.preprocessing.preprocess_scannet import get_thing_semantics
from util.panoptic_quality import panoptic_quality_match, _panoptic_quality_compute
from inference.render_panopli import cluster_segmentwise, cluster


def find_bandwidth(config, debug=False, segmentwise=False, use_dbscan=False):
    output_dir = (Path("runs") / Path(config.experiment))
    print(output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    is_MOS = config.dataset_class == "mos"

    if config.dataset_class == "panopli":
        train_set = PanopLiDataset(
            Path(config.dataset_root), "train", (config.image_dim[0], config.image_dim[1]),
            config.max_depth, overfit=config.overfit, semantics_dir='m2f_semantics', instance_dir='m2f_instance',
            instance_to_semantic_key='m2f_instance_to_semantic', create_seg_data_func=create_segmentation_data_panopli, 
            subsample_frames=config.subsample_frames, do_not_load=True)
    elif config.dataset_class == "mos":
        train_set = MOSDataset(
            Path(config.dataset_root), "train", (config.image_dim[0], config.image_dim[1]),
            config.max_depth, overfit=config.overfit, semantics_dir='detic_semantic', instance_dir='detic_instance',
            instance_to_semantic_key=None, create_seg_data_func=None,
            subsample_frames=config.subsample_frames, do_not_load=True)

    H, W, alpha = config.image_dim[0], config.image_dim[1], 0.65
    train_loader = DataLoader(train_set, shuffle=False, num_workers=0, batch_size=1)
    checkpoint = torch.load(config.resume, map_location="cpu")
    state_dict = checkpoint['state_dict']
    total_classes = len(train_set.segmentation_data.bg_classes) + len(train_set.segmentation_data.fg_classes)
    output_mlp_semantics = torch.nn.Identity() if config.semantic_weight_mode != "softmax" else torch.nn.Softmax(dim=-1)
    model = TensorVMSplit(
        [config.min_grid_dim, config.min_grid_dim, config.min_grid_dim],
        num_semantics_comps=(32, 32, 32), num_instance_comps=(32, 32, 32), num_semantic_classes=total_classes, 
        dim_feature_instance=2*config.max_instances if config.instance_loss_mode=="slow_fast" else config.max_instances,
        output_mlp_semantics=output_mlp_semantics, use_semantic_mlp=config.use_mlp_for_semantics,  use_instance_mlp=config.use_mlp_for_instances,
        use_distilled_features_semantic=config.use_distilled_features_semantic, use_distilled_features_instance=config.use_distilled_features_instance,
        pe_sem=config.pe_sem, pe_ins=config.pe_ins,
        slow_fast_mode=config.instance_loss_mode=="slow_fast", use_proj=config.use_proj)
    renderer = TensoRFRenderer(train_set.scene_bounds, [config.min_grid_dim, config.min_grid_dim, config.min_grid_dim],
                               semantic_weight_mode=config.semantic_weight_mode)
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

    all_thing_features = []
    all_points_semantics = []
    all_points_rgb = []
    all_points_depth = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            batch['rays'] = batch['rays'].squeeze(0).to(device)
            concated_outputs = []
            outputs = []
            # infer semantics and surrogate ids
            for i in range(0, batch['rays'].shape[0], config.chunk):
                out_rgb_, out_semantics_, out_instances_, out_depth_, _, _ = renderer(
                    model, batch['rays'][i: i + config.chunk], config.perturb, train_set.white_bg, False)
                outputs.append([out_rgb_, out_semantics_, out_instances_, out_depth_])
            for i in range(len(outputs[0])):
                concated_outputs.append(torch.cat([outputs[j][i] for j in range(len(outputs))], dim=0))
            p_rgb, p_semantics, p_instances, p_dist = concated_outputs
            p_depth = distance_to_depth(train_set.intrinsics[0], p_dist.view(H, W))
            points_xyz = batch['rays'][...,0:3] + p_dist[...,None] * batch['rays'][...,3:6] # B x 3
            
            all_points_rgb.append(p_rgb)
            all_points_depth.append(p_depth)
            if config.use_delta:
                p_instances = p_instances + points_xyz

            if model.slow_fast_mode:
                p_instances = p_instances[...,0:config.max_instances] # keep fast embeddings only
            
            # create surrogate ids
            p_instances = create_instances_from_semantics(p_instances, p_semantics, train_set.segmentation_data.fg_classes)

            # The following is a HACK:
            # We convert all thing classes to the same class. We do this because otherwise the model can cheat and still get a high PQ score.
            # This is because: if there are different thing segments, the model can predict same instance ID for all of them.
            # Since the tuple (s,i) can be different for each thing segment, the model can still get a high PQ score.
            # To prevent this, we convert all thing classes to the same class. 
            # So, the model is forced to predict a different instance ID for each thing segment to get a high PQ score.
            # if not segmentwise:
            p_semantics = modify_things_to_singleclass_onehot(p_semantics, train_set.segmentation_data.fg_classes)
            all_points_semantics.append(p_semantics)

            all_thing_features.append(p_instances)

    all_thing_features = torch.cat(all_thing_features, dim=0).cpu().numpy()
    np.save(output_dir / "all_thing_features_train.npy", all_thing_features)

    # semantic prediction images
    all_semantic_images = {}
    for i in range(len(train_set)):
        p_semantics = all_points_semantics[i]
        output_semantics_with_invalid = p_semantics.detach().argmax(dim=1)
        name = f"{train_set.all_frame_names[train_set.train_indices[i]]}.png"
        all_semantic_images[name] = output_semantics_with_invalid.reshape(H, W).cpu().numpy().astype(np.uint8)

    # find bandwidth
    best_pq = 0.0
    best_bw = None
    bw_list, pq_list = [], []
    # load all target images
    if not is_MOS:
        all_semantic_target_images, all_instance_target_images = load_all_target_images(
                                                                    list(all_semantic_images.keys()), 
                                                                    Path(cfg.dataset_root, "m2f_semantics"),
                                                                    Path(cfg.dataset_root, "m2f_instance"),
                                                                    config.image_dim,
                                                                )
    else:
        all_semantic_target_images, all_instance_target_images = load_all_target_images_MOS(
                                                                    list(all_semantic_images.keys()), 
                                                                    Path(cfg.dataset_root, "detic_semantic"),
                                                                    Path(cfg.dataset_root, "detic_instance"),
                                                                    config.image_dim,
                                                                )
    
    if not use_dbscan: # find bandwidth for mean-shift
        # sweep_range should be proportional to SQRT(dimensionality of feature space)
        # NOTE: MeanShift doesn't work well with high-dimensional feature spaces, e.g. dim >= 10
        if not is_MOS:
            sweep_range = np.arange(np.sqrt(config.max_instances)/(3.5*25), np.sqrt(config.max_instances)/3.5, np.sqrt(config.max_instances)/3.5/25)
        else:
            sweep_range = np.arange(np.sqrt(config.max_instances)/(3.5*50), np.sqrt(config.max_instances)/3.5, np.sqrt(config.max_instances)/3.5/50)
    else: # find min_cluster_size for HDBSCAN
        if is_MOS:
            sweep_range = np.arange(10, 200, 10) # don't bother sweeping high values for MOS
        else:
            sweep_range = np.arange(250, 3000, 50) # number of values = 55 (?)

    for val in sweep_range:
        try:
            if not segmentwise:
                if use_dbscan:
                    all_points_instances = cluster(
                        all_thing_features, bandwidth=0.15, device=device, cluster_size=val, 
                        num_images=len(train_set), use_dbscan=True)
                else:
                    all_points_instances = cluster(
                        all_thing_features, bandwidth=val, device=device, num_images=len(train_set)) # mean-shift
            else:
                if use_dbscan:
                    all_points_instances, _ = cluster_segmentwise(
                        all_thing_features, all_points_semantics, bandwidth=0.15, device=device,
                        cluster_size=val, num_images=len(train_set), use_dbscan=True)
                else:
                    all_points_instances, _ = cluster_segmentwise(
                        all_thing_features, all_points_semantics, bandwidth=val, device=device,
                        num_images=len(train_set))
        except:
            print(f"Clustering failed for value {val}")
            continue # skip this value if clustering fails
        
        # instance prediction images
        all_instance_images = {}
        for i, _ in enumerate(all_points_instances):
            name = f"{train_set.all_frame_names[train_set.train_indices[i]]}.png"
            p_instances = all_points_instances[i]
            all_instance_images[name] = p_instances.argmax(dim=1).reshape(H, W).cpu().numpy().astype(np.int32)
            
            if debug:
                (output_dir / f"pred_surrogateid_bw_{val}").mkdir(exist_ok=True)
                (output_dir / f"pred_surrogateid_bw_{val}" / "vis_semantics_and_surrogate").mkdir(exist_ok=True)
                p_rgb, p_semantics, p_instances, p_depth = all_points_rgb[i], all_points_semantics[i], all_points_instances[i], all_points_depth[i]
                stack = visualize_panoptic_outputs(p_rgb, p_semantics, p_instances, p_depth, None, None, None, H, W, thing_classes=train_set.segmentation_data.fg_classes, visualize_entropy=False)
                output_semantics_with_invalid = p_semantics.detach().argmax(dim=1)
                grid = (make_grid(stack, value_range=(0, 1), normalize=True, nrow=5).permute((1, 2, 0)).contiguous() * 255).cpu().numpy().astype(np.uint8)
                Image.fromarray(grid).save(output_dir / f"pred_surrogateid_bw_{val}" / "vis_semantics_and_surrogate" / name)

        # NOTE: We are using Mask2Former (or Detic) pseudo-labels to find the bandwidth.
        # We obviously should not use ground-truth labels.
        # Now, since M2F pseudo-labels are not tracked across frames, we use PQ (*NOT* PQ_scene) for sweeping.
        # See below, we use the "per_frame" version.
        if not is_MOS:
            pq, _, _ = MY_calculate_panoptic_quality_per_frame_folders(
                    all_semantic_images,
                    all_instance_images,
                    all_semantic_target_images,
                    all_instance_target_images,
                )
        else:
            pq, _, _ = MY_calculate_panoptic_quality_per_frame_folders_MOS(
                    all_semantic_images,
                    all_instance_images,
                    all_semantic_target_images,
                    all_instance_target_images,
                )
        print(f"bw: {val}, pq: {pq}")
        bw_list.append(val)
        pq_list.append(pq)
        if pq >= best_pq:
            best_pq = pq
            best_bw = val
    
    plt.plot(bw_list, pq_list)
    # draw circle at best bandwidth
    plt.scatter(best_bw, best_pq, s=100, facecolors='none', edgecolors='r')
    plt.xlabel("bandwidth")
    plt.ylabel("panoptic quality")
    plt.title(f"Best bandwidth: {best_bw}, pq: {best_pq}")
    plt.savefig(output_dir / "bandwidth_vs_pq.png")

    print(f"Best bandwidth: {best_bw}, pq: {best_pq}")


def create_instances_from_semantics(instances, semantics, thing_classes):
    stuff_mask = ~torch.isin(semantics.argmax(dim=1), torch.tensor(thing_classes).to(semantics.device))
    padded_instances = torch.ones((instances.shape[0], instances.shape[1] + 1), device=instances.device) * -float('inf')
    padded_instances[:, 1:] = instances
    padded_instances[stuff_mask, 0] = float('inf')
    return padded_instances

def modify_things_to_singleclass_onehot(semantics, thing_classes):
    thing_mask = torch.isin(semantics.argmax(dim=1), torch.tensor(thing_classes).to(semantics.device))
    modified_semantics = torch.zeros_like(semantics)
    modified_semantics[~thing_mask] = semantics[~thing_mask]
    modified_semantics[thing_mask, thing_classes[0]] = 1 # set all thing classes to the first thing class
    return modified_semantics

def modify_things_to_singleclass(img_semantics, thing_classes):
    thing_mask = torch.isin(img_semantics, torch.tensor(thing_classes).to(img_semantics.device))
    modified_semantics = torch.zeros_like(img_semantics)
    modified_semantics[~thing_mask] = img_semantics[~thing_mask]
    modified_semantics[thing_mask] = thing_classes[0] # set all thing classes to the first thing class
    return modified_semantics


def MY_read_and_resize_labels(path, size):
    # sightly modified version of read_and_resize_labels from dataset.preprocessing.preprocess_scannet
    size = (size[1], size[0])
    image = Image.open(path)
    # resize if necessary
    if image.size != size:
        image = image.resize(size, Image.NEAREST)
    return np.array(image)
    

def MY_read_and_resize_labels_npy(path, size):
    # sightly modified version of read_and_resize_labels_npy from dataset.preprocessing.preprocess_scannet
    size = (size[1], size[0])
    image = np.load(path)
    image = Image.fromarray(image.astype(np.int16))
    # resize if necessary
    if image.size != size:
        image = image.resize(size, Image.NEAREST)
    return np.array(image)


def load_all_target_images(filenames, path_target_sem, path_target_inst, image_size):
    train_set = json.loads(Path(path_target_sem.parent / "splits.json").read_text())["train"]
    all_semantic_target_images = {}
    all_instance_target_images = {}
    train_paths = [Path(y) for y in sorted(filenames, key=lambda x: int(Path(x).stem)) if Path(y).stem in train_set]
    for p in tqdm(train_paths, desc="Loading images"):
        all_semantic_target_images[p.name] = MY_read_and_resize_labels((path_target_sem / p.name), image_size)
        all_instance_target_images[p.name] = MY_read_and_resize_labels((path_target_inst / p.name), image_size)
    return all_semantic_target_images, all_instance_target_images


def load_all_target_images_MOS(filenames, path_target_sem, path_target_inst, image_size):
    train_set = sorted([x.stem for x in (path_target_sem).iterdir() if x.name.endswith('.npy')], key=lambda y: int(y) if y.isnumeric() else y)
    train_set = train_set[:int(len(train_set) * 0.8)] # initial "80%" images are used for training. See dataset.many_object_scenes.py
    all_semantic_target_images = {}
    all_instance_target_images = {}
    train_paths = [Path(y) for y in sorted(filenames, key=lambda x: int(Path(x).stem)) if Path(y).stem in train_set]
    for p in tqdm(train_paths, desc="Loading target images"):
        all_semantic_target_images[p.name] = MY_read_and_resize_labels_npy(str(path_target_sem / p.stem)+".npy", image_size)
        all_instance_target_images[p.name] = MY_read_and_resize_labels_npy(str(path_target_inst / p.stem)+".npy", image_size)
    return all_semantic_target_images, all_instance_target_images


# minor modification to the original "calculate_panoptic_quality_per_frame_folders" function
def MY_calculate_panoptic_quality_per_frame_folders(all_semantic_images, all_instance_images, all_semantic_target_images, all_instance_target_images):
    is_thing = get_thing_semantics()
    faulty_gt = [0]
    things = set([i for i in range(len(is_thing)) if is_thing[i]])
    stuff = set([i for i in range(len(is_thing)) if not is_thing[i]])
    val_paths = [Path(y) for y in sorted(all_semantic_images.keys(), key=lambda x: int(Path(x).stem))]
    things_, stuff_, iou_sum_, true_positives_, false_positives_, false_negatives_ = set(), set(), [], [], [], []
    for p in tqdm(val_paths):
        img_target_sem = all_semantic_target_images[p.name]
        valid_mask = ~np.isin(img_target_sem, faulty_gt)
        img_pred_sem = torch.from_numpy(all_semantic_images[p.name][valid_mask]).unsqueeze(-1)
        img_target_sem = torch.from_numpy(img_target_sem[valid_mask]).unsqueeze(-1)

        # modify things to single class in img_target_sem
        img_target_sem = modify_things_to_singleclass(img_target_sem, list(things))

        img_pred_inst = torch.from_numpy(all_instance_images[p.name][valid_mask]).unsqueeze(-1)
        img_target_inst = torch.from_numpy(all_instance_target_images[p.name][valid_mask]).unsqueeze(-1)
        pred_ = torch.cat([img_pred_sem, img_pred_inst], dim=1).reshape(-1, 2)
        target_ = torch.cat([img_target_sem, img_target_inst], dim=1).reshape(-1, 2)
        _things, _stuff, _iou_sum, _true_positives, _false_positives, _false_negatives = panoptic_quality_match(pred_, target_, things, stuff, True)
        things_.union(_things)
        stuff_.union(_stuff)
        iou_sum_.append(_iou_sum)
        true_positives_.append(_true_positives)
        false_positives_.append(_false_positives)
        false_negatives_.append(_false_negatives)
    results = _panoptic_quality_compute(things_, stuff_, torch.cat(iou_sum_, 0), torch.cat(true_positives_, 0), torch.cat(false_positives_, 0), torch.cat(false_negatives_, 0))
    return results["all"]["pq"].item(), results["all"]["sq"].item(), results["all"]["rq"].item()


def MY_calculate_panoptic_quality_per_frame_folders_MOS(all_semantic_images, all_instance_images, all_semantic_target_images, all_instance_target_images):
    is_thing = [False, True] # background, foreground
    
    things = set([i for i in range(len(is_thing)) if is_thing[i]]) # {1}
    stuff = set([i for i in range(len(is_thing)) if not is_thing[i]]) # {0}
    val_paths = [Path(y) for y in sorted(all_semantic_images.keys(), key=lambda x: int(Path(x).stem))]
    things_, stuff_, iou_sum_, true_positives_, false_positives_, false_negatives_ = set(), set(), [], [], [], []
    for p in tqdm(val_paths):
        img_target_sem = all_semantic_target_images[p.name]
        valid_mask = np.ones_like(img_target_sem, dtype=bool) # all pixels are valid
        img_pred_sem = torch.from_numpy(all_semantic_images[p.name][valid_mask]).unsqueeze(-1)
        img_target_sem = torch.from_numpy(img_target_sem[valid_mask]).unsqueeze(-1)

        # modify things to single class in img_target_sem
        img_target_sem = modify_things_to_singleclass(img_target_sem, list(things))

        img_pred_inst = torch.from_numpy(all_instance_images[p.name][valid_mask]).unsqueeze(-1)
        img_target_inst = torch.from_numpy(all_instance_target_images[p.name][valid_mask]).unsqueeze(-1)
        pred_ = torch.cat([img_pred_sem, img_pred_inst], dim=1).reshape(-1, 2)
        target_ = torch.cat([img_target_sem, img_target_inst], dim=1).reshape(-1, 2)
        _things, _stuff, _iou_sum, _true_positives, _false_positives, _false_negatives = panoptic_quality_match(pred_, target_, things, stuff, True)
        things_.union(_things)
        stuff_.union(_stuff)
        iou_sum_.append(_iou_sum)
        true_positives_.append(_true_positives)
        false_positives_.append(_false_positives)
        false_negatives_.append(_false_negatives)
    results = _panoptic_quality_compute(things_, stuff_, torch.cat(iou_sum_, 0), torch.cat(true_positives_, 0), torch.cat(false_positives_, 0), torch.cat(false_negatives_, 0))
    return results["all"]["pq"].item(), results["all"]["sq"].item(), results["all"]["rq"].item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bandwidth search')
    parser.add_argument('--ckpt_path', required=True, type=str,
                        help='path of checkpoint to be used')
    parser.add_argument('--subsample', required=False, type=int,
                        default=5)
    parser.add_argument('--debug', action='store_true',
                        help='visualized images will be saved when debug is True') # default is false
    parser.add_argument('--segmentwise', action='store_true',
                        help='segmentwise clustering') # default is false
    parser.add_argument('--use_dbscan', action='store_true',
                        help='HDBSCAN for clustering') # default is false
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(Path(args.ckpt_path).parents[1] / "config.yaml")
    cfg.resume = args.ckpt_path
    cfg.subsample_frames = args.subsample
    cfg.image_dim = [256, 384]
    if isinstance(cfg.image_dim, int):
        cfg.image_dim = [cfg.image_dim, cfg.image_dim]
    t0 = time.time()
    find_bandwidth(cfg, args.debug, segmentwise=args.segmentwise, use_dbscan=args.use_dbscan)
    t1 = time.time()
    print("Total time for finding bandwidth: ", t1-t0)
