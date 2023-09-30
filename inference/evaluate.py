import sys
import argparse
from pathlib import Path

sys.path.append(".")
from dataset.preprocessing.preprocess_scannet import (
    calculate_iou_folders, 
    calculate_panoptic_quality_folders,
    calculate_iou_folders_MOS, 
    calculate_panoptic_quality_folders_MOS
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='metrics')
    parser.add_argument('--root_path', required=False)
    parser.add_argument('--exp_path', required=False)
    # add flag
    parser.add_argument('--MOS', action='store_true') # evaluating MOS dataset
    args = parser.parse_args()

    print('calculating metrics for ours')
    image_dim = (512, 512)
    if not args.MOS:
        iou = calculate_iou_folders(Path(args.exp_path, "pred_semantics"), Path(args.root_path) / "rs_semantics", image_dim)
        pq, rq, sq = calculate_panoptic_quality_folders(
            Path(args.exp_path, "pred_semantics"), Path(args.exp_path, "pred_surrogateid"),
            Path(args.root_path) / "rs_semantics", Path(args.root_path) / "rs_instance", image_dim)
    else:
        iou = calculate_iou_folders_MOS(Path(args.exp_path, "pred_semantics"), Path(args.root_path) / "semantic", image_dim)
        pq, rq, sq = calculate_panoptic_quality_folders_MOS(
            Path(args.exp_path, "pred_semantics"), Path(args.exp_path, "pred_surrogateid"),
            Path(args.root_path) / "semantic", Path(args.root_path) / "instance", image_dim)
    print(f'[dataset] iou, pq, sq, rq: {iou:.3f}, {pq:.3f}, {sq:.3f}, {rq:.3f}')
    # write metrics to file
    with open(Path(args.exp_path, "metrics.txt"), "w") as f:
        f.write(f'iou, pq, sq, rq: {iou:.3f}, {pq:.3f}, {sq:.3f}, {rq:.3f}')