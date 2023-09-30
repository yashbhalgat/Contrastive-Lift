# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from torch import nn
import torch.nn.functional as F
# from torch_scatter import scatter_mean


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.size_tensor(x[:, :, 1:, :]) + 1e-4
        count_w = self.size_tensor(x[:, :, :, 1:]) + 1e-4
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def size_tensor(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def get_semantic_weights(reweight_classes, fg_classes, num_semantic_classes):
    weights = torch.ones([num_semantic_classes]).float()
    if reweight_classes:
        weights[fg_classes] = 2
    return weights


class SCELoss(torch.nn.Module):

    def __init__(self, alpha, beta, class_weights):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, pred, labels_probabilities):
        # CCE
        ce = self.cross_entropy(pred, labels_probabilities)

        # RCE
        weights = torch.tensor(self.class_weights, device=pred.device).unsqueeze(0)
        pred = F.softmax(pred * weights, dim=1)
        pred = torch.clamp(pred, min=1e-8, max=1.0)
        label_clipped = torch.clamp(labels_probabilities, min=1e-8, max=1.0)

        rce = torch.sum(-1 * (pred * torch.log(label_clipped) * weights), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss


def contrastive_loss(features, instance_labels, temperature):
    bsize = features.size(0)
    masks = instance_labels.view(-1, 1).repeat(1, bsize).eq_(instance_labels.clone())
    masks = masks.fill_diagonal_(0, wrap=False)

    # compute similarity matrix based on Euclidean distance
    distance_sq = torch.pow(features.unsqueeze(1) - features.unsqueeze(0), 2).sum(dim=-1)
    # temperature = 1 for positive pairs and temperature for negative pairs
    temperature = torch.ones_like(distance_sq) * temperature
    temperature = torch.where(masks==1, temperature, torch.ones_like(temperature))

    similarity_kernel = torch.exp(-distance_sq/temperature)
    logits = torch.exp(similarity_kernel)

    p = torch.mul(logits, masks).sum(dim=-1)
    Z = logits.sum(dim=-1)

    prob = torch.div(p, Z)
    prob_masked = torch.masked_select(prob, prob.ne(0))
    loss = -prob_masked.log().sum()/bsize
    return loss


### NOT USED. JUST FOR REFERENCE: Associative Embedding Loss from "Associative Embedding: End-to-End Learning for Joint Detection and Grouping"
# def ae_loss(features, instance_labels, sigma=1.0):
#     # get centroid of each instance
#     # for instance in unique_instances: centroid = mean(features[instance_labels == instance])
#     # verctorized version:
#     unique_instances, inverse_indices = torch.unique(instance_labels, return_inverse=True)
#     centroids = scatter_mean(features, inverse_indices, dim=0, dim_size=unique_instances.shape[0])

#     # Pull loss: pull features towards their instance centroid
#     pull_loss = torch.pow(features - centroids[inverse_indices], 2).sum(dim=-1).mean()

#     # Push loss: push centroids away from each other
#     # for each instance, compute distance to all other instances
#     distances = torch.pow(centroids.unsqueeze(1) - centroids.unsqueeze(0), 2).sum(dim=-1) # (num_instances, num_instances)
#     distances_nondiag = distances[~torch.eye(distances.shape[0], dtype=torch.bool, device=features.device)] # (num_instances * (num_instances - 1))
#     push_loss = torch.exp(-distances_nondiag/sigma).mean()

#     return pull_loss + push_loss
