# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import torch.multiprocessing
import torch
import hydra
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from tabulate import tabulate
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean

sys.path.append(".")
from dataset import get_dataset, get_inconsistent_single_dataset, get_segment_dataset
from model.loss.loss import TVLoss, get_semantic_weights, SCELoss, contrastive_loss
from model.radiance_field.tensoRF import TensorVMSplit
from model.renderer.panopli_tensoRF_renderer import TensoRFRenderer
from util.distinct_colors import DistinctColors
from trainer import create_trainer, get_optimizer_and_scheduler, visualize_panoptic_outputs
from util.metrics import psnr, ConfusionMatrix
from util.panoptic_quality import panoptic_quality
from util.misc import get_parameters_from_state_dict

torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


class TensoRFTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.train_set, self.val_set = get_dataset(config)
        if config.visualized_indices is None:
            config.visualized_indices = list(range(0, len(self.val_set), int(1 / 0.15)))
            if len(config.visualized_indices) < 16:
                config.visualized_indices = list(range(0, len(self.val_set)))[:16]
        config.instance_optimization_epoch = config.instance_optimization_epoch + config.late_semantic_optimization
        config.segment_optimization_epoch = config.segment_optimization_epoch + config.late_semantic_optimization
        self.config = config
        self.current_lambda_dist_reg = 0
        if self.config.segment_grouping_mode != "none":
            self.train_segment_set = get_segment_dataset(self.config)
        self.save_hyperparameters(config)
        total_classes = len(self.train_set.segmentation_data.bg_classes) + len(self.train_set.segmentation_data.fg_classes)
        output_mlp_semantics = torch.nn.Identity() if self.config.semantic_weight_mode != "softmax" else torch.nn.Softmax(dim=-1)
        self.model = TensorVMSplit([config.min_grid_dim, config.min_grid_dim, config.min_grid_dim], num_semantics_comps=(32, 32, 32), num_instance_comps=(32, 32, 32),
                                   num_semantic_classes=total_classes, 
                                   dim_feature_instance=2*self.config.max_instances if self.config.instance_loss_mode=="slow_fast" else self.config.max_instances,
                                   output_mlp_semantics=output_mlp_semantics, use_semantic_mlp=self.config.use_mlp_for_semantics, use_instance_mlp=self.config.use_mlp_for_instances,
                                   use_feature_reg=self.config.use_feature_regularization, # this is NOT actually used
                                   use_distilled_features_semantic=self.config.use_distilled_features_semantic, # this IS used.
                                   use_distilled_features_instance=self.config.use_distilled_features_instance,
                                   pe_sem=self.config.pe_sem, pe_ins=self.config.pe_ins,
                                   slow_fast_mode=self.config.instance_loss_mode=="slow_fast",
                                   use_proj=self.config.use_proj,
                                   )
        self.renderer = TensoRFRenderer(self.train_set.scene_bounds, [config.min_grid_dim, config.min_grid_dim, config.min_grid_dim],
                                        semantic_weight_mode=self.config.semantic_weight_mode, stop_semantic_grad=config.stop_semantic_grad,
                                        feature_stop_grad=config.feature_stop_grad)
        semantic_weights = get_semantic_weights(config.reweight_fg, self.train_set.segmentation_data.fg_classes, self.train_set.segmentation_data.num_semantic_classes)
        semantic_weights[0] = config.weight_class_0
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.loss_feat = torch.nn.L1Loss(reduction='mean')
        self.tv_regularizer = TVLoss()
        if not self.config.use_symmetric_ce:
            self.loss_semantics = torch.nn.CrossEntropyLoss(reduction='none', weight=semantic_weights)
        else:
            self.loss_semantics = SCELoss(self.config.ce_alpha, self.config.ce_beta, semantic_weights)
        self.loss_instances_cluster = torch.nn.CrossEntropyLoss(reduction='none')
        self.instance_loss_mode = self.config.instance_loss_mode # linear_assignment, contrastive, slow_fast
        self.use_DINO_style = self.config.use_DINO_style # if DINO style learning is used for instances
        # if contrastive loss used for instance clustering
        self.temperature = self.config.temperature
        self.use_delta = self.config.use_delta # use features or (points_xyz + features)

        self.output_dir_result_images = Path(f'runs/{self.config.experiment}/images')
        self.output_dir_result_images.mkdir(exist_ok=True)
        self.output_dir_result_clusters = Path(f'runs/{self.config.experiment}/instance_clusters')
        self.output_dir_result_clusters.mkdir(exist_ok=True)
        self.automatic_optimization = False
        self.distinct_colors = DistinctColors()

        if self.instance_loss_mode=="slow_fast" and self.use_DINO_style:
            self.center = torch.zeros(1, self.config.max_instances if not self.config.use_proj else 32).to(self.device)
            self.center_momentum = 0.9
            
        self.validation_step_outputs = []

    def configure_optimizers(self):
        params = self.model.get_optimizable_parameters(self.config.lr * 20, self.config.lr, weight_decay=self.config.weight_decay)
        optimizer, scheduler = get_optimizer_and_scheduler(params, self.config, betas=(0.9, 0.99))
        param_instance = self.model.get_optimizable_instance_parameters(self.config.lr * 20, self.config.lr, using_DINO=self.use_DINO_style)
        optimizer_instance, scheduler_instance = get_optimizer_and_scheduler(param_instance, self.config, betas=(0.9, 0.999))
        return [optimizer, optimizer_instance], [scheduler, scheduler_instance]

    def forward(self, rays, is_train):
        B = rays.shape[0]
        out_rgb, out_semantics, out_instances, out_depth, out_regfeat, out_dist_regularizer = [], [], [], [], [], []
        for i in range(0, B, self.config.chunk):
            out_rgb_, out_semantics_, out_instances_, out_depth_, out_regfeat_, out_dist_reg_ =\
                self.renderer(self.model, rays[i: i + self.config.chunk], self.config.perturb, self.train_set.white_bg, is_train)
            out_rgb.append(out_rgb_)
            out_semantics.append(out_semantics_)
            out_instances.append(out_instances_)
            out_regfeat.append(out_regfeat_)
            out_depth.append(out_depth_)
            out_dist_regularizer.append(out_dist_reg_.unsqueeze(0))
        out_rgb = torch.cat(out_rgb, 0)
        out_instances = torch.cat(out_instances, 0)
        out_depth = torch.cat(out_depth, 0)
        out_semantics = torch.cat(out_semantics, 0)
        out_regfeat = torch.cat(out_regfeat, 0)
        out_dist_regularizer = torch.mean(torch.cat(out_dist_regularizer, 0))
        return out_rgb, out_semantics, out_instances, out_depth, out_regfeat, out_dist_regularizer

    def forward_instance(self, rays, is_train):
        B = rays.shape[0]
        out_feats_instance = []
        points_xyz = []
        for i in range(0, B, self.config.chunk):
            batch_rays = rays[i: i + self.config.chunk]
            out_feats_instance_, points_xyz_ = self.renderer.forward_instance_feature(self.model, batch_rays, self.config.perturb, is_train)
            out_feats_instance.append(out_feats_instance_)
            points_xyz.append(points_xyz_)
        out_feats_instance = torch.cat(out_feats_instance, 0)
        points_xyz = torch.cat(points_xyz, 0)
        return out_feats_instance, points_xyz

    def forward_segments(self, rays, is_train):
        B = rays.shape[0]
        out_feats_segments = []
        for i in range(0, B, self.config.chunk_segment):
            batch_rays = rays[i: i + self.config.chunk_segment]
            out_feats_segments_ = self.renderer.forward_segment_feature(self.model, batch_rays, self.config.perturb, is_train)
            out_feats_segments.append(out_feats_segments_)
        out_feats_segments = torch.cat(out_feats_segments, 0)
        return out_feats_segments

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        schedulers = self.lr_schedulers()
        if not self.config.optimize_instance_only:
            opts[0].zero_grad(set_to_none=True)
            rays, rgbs, semantics, probs, confs, masks, feats = batch[0]['rays'], batch[0]['rgbs'], batch[0]['semantics'], batch[0]['probabilities'], batch[0]['confidences'], batch[0]['mask'], batch[0]['feats']
            output_rgb, output_semantics, _, _, output_feats, loss_dist_reg = self(rays, True)
            output_rgb[~masks, :] = 0
            rgbs[~masks, :] = 0
            confs[~masks] = 0
            loss = torch.zeros(1, device=batch[0]['rays'].device, requires_grad=True)
            if self.config.lambda_rgb > 0:
                loss_rgb = self.loss(output_rgb, rgbs)
                loss_tv = self.model.total_tv_loss(self.tv_regularizer, self.config, self.current_epoch)
                
                loss_feat = torch.zeros_like(loss_tv)
                if self.config.use_distilled_features_semantic or self.config.use_distilled_features_instance: # this means the distilled feature grid exists
                    if self.current_epoch <= self.config.feature_optimization_end_epoch: # this means we are still optimizing the features
                        loss_tv_distilled_features = self.model.tv_loss_distilled_features(self.tv_regularizer)
                        loss_tv += loss_tv_distilled_features * self.config.lambda_tv_distilled_features
                        loss_feat = self.loss_feat(output_feats, feats).mean()

                loss = self.config.lambda_rgb * (loss_rgb + loss_tv + loss_dist_reg * self.current_lambda_dist_reg + loss_feat * self.config.lambda_feat)
                self.log("train/loss_rgb", loss_rgb, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
                self.log("train/loss_feat", loss_feat, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
                self.log("train/loss_dist_regularizer", loss_dist_reg, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            
            loss_semantics = torch.zeros(1, device=self.device, requires_grad=True)
            if self.current_epoch >= self.config.late_semantic_optimization:
                if self.config.probabilistic_ce_mode == "TTAConf":
                    loss_semantics = (self.loss_semantics(output_semantics, probs) * confs).mean()
                elif self.config.probabilistic_ce_mode == "NoTTAConf":
                    loss_semantics = (self.loss_semantics(output_semantics, semantics) * confs).mean()
                else:
                    loss_semantics = self.loss_semantics(output_semantics, semantics).mean()

            loss_segment_clustering = torch.zeros(1, device=self.device, requires_grad=True)
            if self.config.segment_grouping_mode != "none" and self.current_epoch >= self.config.segment_optimization_epoch:
                batch[2]["rays"] = torch.cat(batch[2]['rays'], dim=0)
                batch[2]['group'] = torch.cat(batch[2]['group'], dim=0)
                batch[2]['confidences'] = torch.cat(batch[2]['confidences'], dim=0)
                semantic_features = self.forward_segments(batch[2]["rays"], True)
                batch_target_mean = torch.zeros(self.config.batch_size_segments, semantic_features.shape[-1], device=semantic_features.device)
                scatter_mean(semantic_features, batch[2]['group'], 0, batch_target_mean)
                
                target = batch_target_mean[batch[2]['group'], :].argmax(-1)
                loss_segment_clustering = (self.loss_semantics(semantic_features, target) * batch[2]['confidences']).mean()
                self.log("train/loss_segment", loss_segment_clustering, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            if self.current_epoch >= self.config.late_semantic_optimization:
                loss = loss + self.config.lambda_semantics * loss_semantics + self.config.lambda_semantics * self.config.lambda_segment * loss_segment_clustering
            self.manual_backward(loss)
            opts[0].step()
            metric_psnr = psnr(output_rgb, rgbs)
            self.log("train/loss_semantics", loss_semantics, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            self.log("lr", opts[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/psnr", metric_psnr, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            output_semantics_without_invalid = output_semantics.detach().argmax(dim=1)
            output_semantics_without_invalid[semantics == 0] = 0
            train_cm = ConfusionMatrix(num_classes=self.model.num_semantic_classes, ignore_class=[0])
            metric_iou = train_cm.add_batch(output_semantics_without_invalid.cpu().numpy(), semantics.cpu().numpy(), return_miou=True)
            self.log("train/iou", metric_iou, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        if self.current_epoch >= self.config.instance_optimization_epoch:
            opts[1].zero_grad(set_to_none=True)
            loss_instance_clustering = torch.zeros(1, device=self.device, requires_grad=True)
            for img_idx in range(len(batch[1]['rays'])):
                instance_features, points_xyz = self.forward_instance(batch[1]['rays'][img_idx], True)
                loss_instance_clustering_ = self.calculate_instance_clustering_loss(
                    batch[1]['instances'][img_idx], instance_features, batch[1]['confidences'][img_idx],
                    points_xyz)
                loss_instance_clustering = loss_instance_clustering + loss_instance_clustering_
            instance_features.retain_grad()
            self.manual_backward(loss_instance_clustering)
            opts[1].step()
            self.log("train/loss_clustering", loss_instance_clustering, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            # self.log(f"train/grad_losscluster", (instance_features.grad.var(0)/(instance_features.grad.mean(0)+1e-15)).sum(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        
        # Scheduler step at the end of the epoch since we are using manual optimization
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            schedulers[0].step()
            schedulers[1].step()

    def calculate_instance_clustering_loss(self, labels_gt, instance_features, confidences, points_xyz=None):
        '''
        mode: linear_assignment, contrastive, ae_loss, slow_fast
        delta: 
            if False, contrastive loss applied over features as usual. 
            if True, features should be 3D and contrastive loss applied over (points_xyz + features)
        ''' 
        if self.instance_loss_mode == "linear_assignment":
            virtual_gt_labels = self.create_virtual_gt_with_linear_assignment(labels_gt, instance_features)
            predicted_labels = instance_features.argmax(dim=-1)
            if torch.any(virtual_gt_labels != predicted_labels):  # should never reinforce correct labels
                return (self.loss_instances_cluster(instance_features, virtual_gt_labels) * confidences).mean()
            return torch.tensor(0., device=self.device, requires_grad=True)
        if self.instance_loss_mode == "contrastive":
            if self.use_delta:
                assert instance_features.shape[-1] == 3, "delta mode only works with 3D features"
                instance_features = points_xyz + instance_features
            loss_ = contrastive_loss(instance_features, labels_gt, self.temperature)
            if self.use_delta:
                loss_ += 0.1 * torch.norm(instance_features-points_xyz, dim=-1).mean() # penalize norm of delta
            return loss_
        if self.instance_loss_mode == "ae_loss":
            if self.use_delta:
                assert instance_features.shape[-1] == 3, "delta mode only works with 3D features"
                instance_features = points_xyz + instance_features
            return ae_loss(instance_features, labels_gt, sigma=self.temperature)
        if self.instance_loss_mode == "slow_fast":
            # EMA update of slow network; done before everything else
            ema_momentum = 0.9 # CONSTANT MOMENTUM
            self.ema_update_slownet(self.model.render_instance_mlp.slow_mlp, self.model.render_instance_mlp.mlp, ema_momentum)

            fast_features, slow_features = instance_features.split(
                [self.model.dim_feature_instance//2, self.model.dim_feature_instance//2], dim=-1)
            if self.config.use_proj:
                fast_projections, slow_projections = self.model.proj_layer(fast_features, slow_features)
                # NOTE the inputs in line below
                self.ema_update_slownet(self.model.proj_layer.slow_proj, self.model.proj_layer.fast_proj, ema_momentum)
            else:
                fast_projections, slow_projections = fast_features, slow_features # no projection layer
            slow_projections = slow_projections.detach() # no gradient for slow projections

            # sample two random batches from the current batch
            fast_mask = torch.zeros_like(labels_gt).bool()
            fast_mask[:labels_gt.shape[0] // 2] = True
            slow_mask = ~fast_mask # non-overlapping masks for slow and fast models
            
            ## compute centroids
            slow_centroids = []
            fast_labels, slow_labels = torch.unique(labels_gt[fast_mask]), torch.unique(labels_gt[slow_mask])
            for l in slow_labels:
                mask_ = torch.logical_and(slow_mask, labels_gt==l) #.unsqueeze(-1)
                slow_centroids.append(slow_projections[mask_].mean(dim=0))
            slow_centroids = torch.stack(slow_centroids)

            # DEBUG edge case:
            if len(fast_labels) == 0 or len(slow_labels) == 0:
                print("Length of fast labels", len(fast_labels), "Length of slow labels", len(slow_labels))
                # This happens when labels_gt of shape 1
                return torch.tensor(0.0, device=instance_features.device)

            ### Concentration loss
            intersecting_labels = fast_labels[torch.where(torch.isin(fast_labels, slow_labels))] # [num_centroids]
            loss = 0
            for l in intersecting_labels:
                mask_ = torch.logical_and(fast_mask, labels_gt==l)
                centroid_ = slow_centroids[slow_labels==l] # [1, d]
                # distance between fast features and slow centroid
                dist_sq = torch.pow(fast_projections[mask_] - centroid_, 2).sum(dim=-1) # [num_points]
                loss += -1.0 * (torch.exp(-dist_sq / 1.0) * confidences[mask_]).mean()
            if intersecting_labels.shape[0] > 0: 
                loss /= intersecting_labels.shape[0]
            
            ### Contrastive loss
            label_matrix = labels_gt[fast_mask].unsqueeze(1) == labels_gt[slow_mask].unsqueeze(0) # [num_points1, num_points2]
            similarity_matrix = torch.exp(-torch.cdist(fast_projections[fast_mask], slow_projections[slow_mask], p=2) / 1.0) # [num_points1, num_points2]
            logits = torch.exp(similarity_matrix)
            # compute loss
            prob = torch.mul(logits, label_matrix).sum(dim=-1) / logits.sum(dim=-1)
            prob_masked = torch.masked_select(prob, prob.ne(0))
            loss += -torch.log(prob_masked).mean()
            return loss
            
        print("Unknown instance loss mode")
        raise NotImplementedError

    @torch.no_grad()
    def update_center(self, slow_features):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(slow_features, dim=0, keepdim=True) / slow_features.shape[0] # for centering slow features
        # ema update
        self.center = self.center.to(slow_features.device)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def ema_update_slownet(self, slownet, fastnet, momentum):
        # EMA update for the teacher
        with torch.no_grad():
            for param_q, param_k in zip(fastnet.parameters(), slownet.parameters()):
                param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    @torch.no_grad()
    def create_virtual_gt_with_linear_assignment(self, labels_gt, predicted_scores):
        labels = sorted(torch.unique(labels_gt).cpu().tolist())[:predicted_scores.shape[-1]]
        predicted_probabilities = torch.softmax(predicted_scores, dim=-1)
        cost_matrix = np.zeros([len(labels), predicted_probabilities.shape[-1]])
        for lidx, label in enumerate(labels):
            cost_matrix[lidx, :] = -(predicted_probabilities[labels_gt == label, :].sum(dim=0) / ((labels_gt == label).sum() + 1e-4)).cpu().numpy()
        assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
        new_labels = torch.zeros_like(labels_gt)
        for aidx, lidx in enumerate(assignment[0]):
            new_labels[labels_gt == labels[lidx]] = assignment[1][aidx]
        return new_labels

    def calculate_segment_clustering_loss(self, sem_features, confidences):
        target = torch.mean(torch.softmax(sem_features, dim=-1), dim=0).detach().unsqueeze(0).expand(sem_features.shape[0], -1)
        if self.config.segment_grouping_mode == "argmax_noconf":
            return self.loss_semantics(sem_features, target.argmax(-1)).mean()
        if self.config.segment_grouping_mode == "argmax_conf":
            return (self.loss_semantics(sem_features, target.argmax(-1)) * confidences).mean()
        if self.config.segment_grouping_mode == "prob_noconf":
            return self.loss_semantics(sem_features, target).mean()
        if self.config.segment_grouping_mode == "prob_conf":
            return (self.loss_semantics(sem_features, target) * confidences).mean()
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        rays, rgbs, semantics, instances, mask = batch['rays'].squeeze(), batch['rgbs'].squeeze(), batch['semantics'].squeeze(), batch['instances'].squeeze(), batch['mask'].squeeze()
        rs_semantics, rs_instances = batch['rs_semantics'].squeeze(), batch['rs_instances'].squeeze()
        probs, confs = batch['probabilities'].squeeze(), batch['confidences'].squeeze()
        output_rgb, output_semantics, output_instances, _output_depth, _, _ = self(rays, False)
        output_rgb[torch.logical_not(mask), :] = 0
        rgbs[torch.logical_not(mask), :] = 0
        loss = self.loss(output_rgb, rgbs)
        metric_psnr = psnr(output_rgb, rgbs)
        self.log("val/loss_rgb", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/psnr", metric_psnr, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        output_semantics[torch.logical_not(mask)] = 0
        if self.config.probabilistic_ce_mode == "TTAConf":
            loss_semantics = (self.loss_semantics(output_semantics, probs) * confs).mean()
        elif self.config.probabilistic_ce_mode == "NoTTAConf":
            loss_semantics = (self.loss_semantics(output_semantics, semantics) * confs).mean()
        else:
            loss_semantics = self.loss_semantics(output_semantics, semantics).mean()
        output_semantics_without_invalid = output_semantics.detach().argmax(dim=1)
        output_semantics_without_invalid[semantics == 0] = 0
        val_cm = ConfusionMatrix(num_classes=self.model.num_semantic_classes, ignore_class=[0])
        metric_iou = val_cm.add_batch(output_semantics_without_invalid.cpu().numpy(), semantics.cpu().numpy(), return_miou=True)
        pano_pred = torch.cat([output_semantics_without_invalid.unsqueeze(1), output_instances.argmax(dim=1).unsqueeze(1)], dim=1)
        pano_target = torch.cat([semantics.unsqueeze(1), instances.unsqueeze(1)], dim=1)
        metric_pq, metric_sq, metric_rq = panoptic_quality(pano_pred, pano_target, self.train_set.things_filtered, self.train_set.stuff_filtered, allow_unknown_preds_category=True)
        self.log("val/iou", metric_iou, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/pq", metric_pq, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/sq", metric_sq, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/rq", metric_rq, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/loss_semantics", loss_semantics, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        val_rs_cm = ConfusionMatrix(num_classes=self.model.num_semantic_classes, ignore_class=list(self.train_set.faulty_classes))
        output_semantics_with_invalid = output_semantics.detach().argmax(dim=1)
        metric_rs_iou = val_rs_cm.add_batch(output_semantics_with_invalid.cpu().numpy(), rs_semantics.cpu().numpy(), return_miou=True)
        pano_rs_pred = torch.cat([output_semantics_with_invalid.unsqueeze(1), output_instances.argmax(dim=1).unsqueeze(1)], dim=1)
        pano_rs_target = torch.cat([rs_semantics.unsqueeze(1), rs_instances.unsqueeze(1)], dim=1)
        metric_rs_pq, metric_rs_sq, metric_rs_rq = panoptic_quality(pano_rs_pred, pano_rs_target, self.train_set.things_filtered, self.train_set.stuff_filtered, allow_unknown_preds_category=True)
        self.log("val_rs/iou", metric_rs_iou, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_rs/pq", metric_rs_pq, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_rs/sq", metric_rs_sq, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_rs/rq", metric_rs_rq, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        metrics_data = {'loss_rgb': loss.item(), 'loss_sem': loss_semantics.item(), 'psnr': metric_psnr.item(),
                'iou': metric_iou.item(), 'pq': metric_pq.item(), 'sq': metric_sq.item(), 'rq': metric_rq.item(),
                'rs_iou': metric_rs_iou.item(), 'rs_pq': metric_rs_pq.item(), 'rs_sq': metric_rs_sq.item(), 'rs_rq': metric_rs_rq.item()}
        self.validation_step_outputs.append(metrics_data)
        return metrics_data

    @rank_zero_only
    def on_validation_epoch_end(self):
        print()
        val_step_outputs = self.validation_step_outputs
        table = [('loss_rgb', 'loss_sem', 'psnr', 'iou', 'pq', 'sq', 'rq', 'rs_iou', 'rs_pq', 'rs_sq', 'rs_rq'), ]
        table.append(tuple([np.array([val_step_outputs[i][key] for i in range(len(val_step_outputs))]).mean() for key in table[0]]))
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        H, W = self.config.image_dim[0], self.config.image_dim[1]
        (self.output_dir_result_clusters / f"{self.global_step:06d}").mkdir(exist_ok=True)
        # self.renderer.export_instance_clusters(self.model, self.output_dir_result_clusters / f"{self.global_step:06d}")
        for batch_idx, batch in enumerate(self.val_dataloader()):
            if batch_idx in self.config.visualized_indices:
                rays, rgbs, semantics, instances = batch['rays'].squeeze().to(self.device), batch['rgbs'].squeeze().to(self.device), \
                                                   batch['semantics'].squeeze().to(self.device), batch['instances'].squeeze().to(self.device)
                rs_semantics, rs_instances = batch['rs_semantics'].squeeze().to(self.device), batch['rs_instances'].squeeze().to(self.device)
                mask = batch['mask'].squeeze().to(self.device)
                output_rgb, output_semantics, output_instances, output_depth, _, _ = self(rays, False)
                output_rgb[torch.logical_not(mask), :] = 0
                output_semantics[torch.logical_not(mask)] = 0
                output_instances[torch.logical_not(mask)] = 0
                rgbs[torch.logical_not(mask), :] = 0
                stack = visualize_panoptic_outputs(output_rgb, output_semantics, output_instances, output_depth, rgbs, rs_semantics, rs_instances, H, W, thing_classes=self.train_set.segmentation_data.fg_classes,
                                                   m2f_semantics=semantics, m2f_instances=instances)
                save_image(stack, self.output_dir_result_images / f"{self.global_step:06d}_{batch_idx:04d}.jpg", value_range=(0, 1), nrow=5, normalize=True)
                if self.config.logger == 'wandb':
                    self.logger.log_image(key=f"images/{batch_idx:04d}", images=[make_grid(stack, value_range=(0, 1), nrow=5, normalize=True)])
                else:
                    self.logger.experiment.add_image(f'visuals/{batch_idx:04d}', make_grid(stack, value_range=(0, 1), nrow=5, normalize=True), global_step=self.global_step)
        self.validation_step_outputs = []

    def train_dataloader(self):
        loaders = {
            0: DataLoader(self.train_set, self.config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=self.config.num_workers)
        }
        train_instance_set = get_inconsistent_single_dataset(self.config)
        assert len(train_instance_set) > 0, "Warning: Empty instance dataset"
        loaders[1] = DataLoader(train_instance_set, self.config.batch_size_contrastive, shuffle=True, drop_last=True, collate_fn=train_instance_set.collate_fn, num_workers=0)
        if self.config.segment_grouping_mode != "none":
            loaders[2] = DataLoader(self.train_segment_set, self.config.batch_size_segments, shuffle=False, drop_last=True, collate_fn=self.train_segment_set.collate_fn, num_workers=0)
        return loaders

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    def on_train_epoch_start(self):
        self.current_lambda_dist_reg = self.config.lambda_dist_reg * (1 - math.exp(-0.25 * self.current_epoch))
        if self.current_epoch in self.config.bbox_aabb_reset_epochs:
            self.renderer.update_bbox_aabb_and_shrink(self.model)
        if self.current_epoch in self.config.grid_upscale_epochs:
            num_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(self.config.min_grid_dim**3), np.log(self.config.max_grid_dim**3), len(self.config.grid_upscale_epochs)+1))).long()).tolist()[1:]
            target_num_voxels = num_voxel_list[self.config.grid_upscale_epochs.index(self.current_epoch)]
            target_resolution = self.renderer.get_target_resolution(target_num_voxels)
            self.config.weight_decay = 0
            self.model.upsample_volume_grid(target_resolution)
            self.renderer.update_step_size(target_resolution)
            self.trainer.strategy.setup_optimizers(self.trainer)
        if self.config.segment_grouping_mode != "none" and self.current_epoch >= self.config.segment_optimization_epoch:
            self.train_segment_set.enabled = True

    def on_load_checkpoint(self, checkpoint):
        for epoch in self.config.grid_upscale_epochs[::-1]:
            if checkpoint['epoch'] >= epoch:
                grid_dim = checkpoint["state_dict"]["renderer.grid_dim"].cpu()
                self.model.upsample_volume_grid(grid_dim)
                self.renderer.bbox_aabb = checkpoint["state_dict"]["renderer.bbox_aabb"]
                self.renderer.update_step_size(grid_dim)
                self.config.weight_decay = 0
                self.trainer.strategy.setup_optimizers(self.trainer)
                break


@hydra.main(config_path='../config', config_name='config', version_base='1.2')
def main(config):
    if config.template.dataset_class=="panopli":
        name = "PanopLi"
    elif config.template.dataset_class=="mos":
        name = "MOS"
    trainer = create_trainer(name, config.template)
    model = TensoRFTrainer(config.template)

    if config.template.resume:
        trainer.fit(model, ckpt_path=config.template.resume)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
