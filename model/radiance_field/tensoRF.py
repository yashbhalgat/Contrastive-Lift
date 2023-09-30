# MIT License
#
# Copyright (c) 2022 Anpei Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from torch import nn
import torch.nn.functional as F

from util.misc import get_parameters_from_state_dict, trunc_normal_


class TensorVMSplit(nn.Module):

    def __init__(self, grid_dim, num_density_comps=(16, 16, 16), num_appearance_comps=(48, 48, 48), num_semantics_comps=None, num_instance_comps=None,
                 dim_appearance=27, dim_semantics=27, dim_instances=27, splus_density_shift=-10, pe_view=2, pe_feat=2, 
                 dim_mlp_color=128, dim_mlp_semantics=128, dim_mlp_instance=256, num_semantic_classes=0, dim_feature_instance=None,
                 output_mlp_semantics=torch.nn.Softmax(dim=-1),
                 use_semantic_mlp=False, use_instance_mlp=False, use_feature_reg=False,
                 use_distilled_features_semantic=False, use_distilled_features_instance=False, num_feature_comps=(48, 48, 48),
                 pe_sem=0, pe_ins=0,
                 slow_fast_mode=False, use_proj=False):
        super().__init__()
        self.num_density_comps = num_density_comps
        self.num_appearance_comps = num_appearance_comps
        self.num_semantics_comps = num_semantics_comps
        self.num_instance_comps = num_instance_comps
        self.dim_appearance = dim_appearance
        self.dim_semantics = dim_semantics
        self.dim_instances = dim_instances
        self.dim_feature_instance = dim_feature_instance
        ins_out_channels = dim_feature_instance//2 if slow_fast_mode else dim_feature_instance
        self.num_semantic_classes = num_semantic_classes
        self.splus_density_shift = splus_density_shift
        self.use_semantic_mlp = use_semantic_mlp
        self.use_instance_mlp = use_instance_mlp
        self.slow_fast_mode = slow_fast_mode
        self.use_proj = use_proj
        self.use_feature_reg = use_feature_reg and use_semantic_mlp
        self.pe_view, self.pe_feat = pe_view, pe_feat
        self.dim_mlp_color = dim_mlp_color
        self.matrix_mode = [[0, 1], [0, 2], [1, 2]]
        self.vector_mode = [2, 1, 0]
        self.density_plane, self.density_line = self.init_one_svd(self.num_density_comps, grid_dim, 0.1)
        self.appearance_plane, self.appearance_line = self.init_one_svd(self.num_appearance_comps, grid_dim, 0.1)
        self.appearance_basis_mat = torch.nn.Linear(sum(self.num_appearance_comps), self.dim_appearance, bias=False)
        self.render_appearance_mlp = MLPRenderFeature(dim_appearance, 3, pe_view, pe_feat, dim_mlp_color)
        self.semantic_plane, self.semantic_line, self.semantic_basis_mat = None, None, None
        self.instance_plane, self.instance_line, self.instance_basis_mat = None, None, None
        self.render_semantic_mlp, self.render_instance_mlp = None, None
        if self.dim_feature_instance is not None:
            if self.num_instance_comps is not None and not use_instance_mlp:
                self.instance_plane, self.instance_line = self.init_one_svd(self.num_instance_comps, grid_dim, 0.1)
                self.instance_basis_mat = torch.nn.Linear(sum(self.num_instance_comps), self.dim_instances, bias=False)
                self.render_instance_mlp = MLPRenderInstanceFeature(self.dim_instances, ins_out_channels, num_mlp_layers=3, dim_mlp=dim_mlp_instance, output_activation=torch.nn.Identity(), use_features=use_distilled_features_instance,
                                                                    slow_fast_mode=slow_fast_mode)
            elif use_instance_mlp:
                self.render_instance_mlp = MLPRenderInstanceFeature(3, ins_out_channels, pe_feat=pe_ins, num_mlp_layers=4, dim_mlp=dim_mlp_instance, output_activation=torch.nn.Identity(), use_features=use_distilled_features_instance,
                                                                    slow_fast_mode=slow_fast_mode)
        if self.num_semantics_comps is not None and not use_semantic_mlp:
            self.semantic_plane, self.semantic_line = self.init_one_svd(self.num_semantics_comps, grid_dim, 0.1)
            self.semantic_basis_mat = torch.nn.Linear(sum(self.num_semantics_comps), self.dim_semantics, bias=False)
            # self.render_semantic_mlp = MLPRenderFeature(self.dim_semantics, num_semantic_classes, 0, 0, dim_mlp_semantics, output_activation=output_mlp_semantics)
            self.render_semantic_mlp = MLPRenderSemanticFeature(self.dim_semantics, num_semantic_classes, num_mlp_layers=3, dim_mlp=dim_mlp_semantics, output_activation=output_mlp_semantics, use_features=use_distilled_features_semantic)
        elif use_semantic_mlp:
            self.render_semantic_mlp = (MLPRenderSemanticFeature if not self.use_feature_reg else MLPRenderSemanticFeatureWithRegularization)(3, num_semantic_classes, pe_feat=pe_sem, output_activation=output_mlp_semantics, use_features=use_distilled_features_semantic)

        self.use_distilled_features_semantic = use_distilled_features_semantic
        self.use_distilled_features_instance = use_distilled_features_instance
        self.num_feature_comps = num_feature_comps
        self.feature_plane, self.feature_line, self.feature_basis_mat, self.render_feature_mlp = None, None, None, None
        if use_distilled_features_semantic or use_distilled_features_instance: # this means feature grid will be created/used
            self.feature_plane, self.feature_line = self.init_one_svd(self.num_feature_comps, grid_dim, 0.1)
            self.feature_basis_mat = torch.nn.Linear(sum(self.num_feature_comps), 96, bias=False)
            self.render_feature_mlp = MLPRenderFeature(96, 64, 0, 0, 256, output_activation=torch.nn.Tanh())
        
        if use_proj:
            self.proj_layer = SlowFastProjLayer(ins_out_channels, 32)

    def init_one_svd(self, n_components, grid_resolution, scale):
        plane_coef, line_coef = [], []
        for i in range(len(self.vector_mode)):
            vec_id = self.vector_mode[i]
            mat_id_0, mat_id_1 = self.matrix_mode[i]
            plane_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_components[i], grid_resolution[mat_id_1], grid_resolution[mat_id_0])), requires_grad=True))
            line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_components[i], grid_resolution[vec_id], 1)), requires_grad=True))
        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)

    def get_coordinate_plane_line(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matrix_mode[0]], xyz_sampled[..., self.matrix_mode[1]], xyz_sampled[..., self.matrix_mode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vector_mode[0]], xyz_sampled[..., self.vector_mode[1]], xyz_sampled[..., self.vector_mode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        return coordinate_plane, coordinate_line

    def compute_density_without_activation(self, xyz_sampled):
        coordinate_plane, coordinate_line = self.get_coordinate_plane_line(xyz_sampled)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature + self.splus_density_shift

    def compute_density(self, xyz_sampled):
        return F.softplus(self.compute_density_without_activation(xyz_sampled))

    def compute_feature(self, xyz_sampled, feature_plane, feature_line, basis_mat):
        coordinate_plane, coordinate_line = self.get_coordinate_plane_line(xyz_sampled)
        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(feature_plane)):
            plane_coef_point.append(F.grid_sample(feature_plane[idx_plane], coordinate_plane[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(feature_line[idx_plane], coordinate_line[[idx_plane]], align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        return basis_mat((plane_coef_point * line_coef_point).T)

    def compute_appearance_feature(self, xyz_sampled):
        return self.compute_feature(xyz_sampled, self.appearance_plane, self.appearance_line, self.appearance_basis_mat)
    
    def compute_distilled_feature(self, xyz_sampled):
        return self.compute_feature(xyz_sampled, self.feature_plane, self.feature_line, self.feature_basis_mat)

    def compute_semantic_feature(self, xyz_sampled):
        if self.use_semantic_mlp:
            return xyz_sampled
        return self.compute_feature(xyz_sampled, self.semantic_plane, self.semantic_line, self.semantic_basis_mat)

    def render_instance_grid(self, xyz_sampled):
        retval = F.one_hot(F.grid_sample(self.instance_grid, xyz_sampled.unsqueeze(0).unsqueeze(0).unsqueeze(0), align_corners=True, padding_mode="border", mode='nearest').squeeze().long(), num_classes=self.dim_feature_instance).float()
        retval = torch.log(retval + 1e-8)
        return retval

    def compute_instance_feature(self, xyz_sampled):
        if self.use_instance_mlp:
            return xyz_sampled
        # return self.render_instance_mlp(xyz_sampled)
        return self.compute_feature(xyz_sampled, self.instance_plane, self.instance_line, self.instance_basis_mat)

    @torch.no_grad()
    def shrink(self, t_l, b_r):
        for i in range(len(self.vector_mode)):
            mode0 = self.vector_mode[i]
            self.density_line[i] = torch.nn.Parameter(self.density_line[i].data[..., t_l[mode0]:b_r[mode0], :])
            self.appearance_line[i] = torch.nn.Parameter(self.appearance_line[i].data[..., t_l[mode0]:b_r[mode0], :])
            if self.semantic_line is not None:
                self.semantic_line[i] = torch.nn.Parameter(self.semantic_line[i].data[..., t_l[mode0]:b_r[mode0], :])
            if self.instance_line is not None:
                self.instance_line[i] = torch.nn.Parameter(self.instance_line[i].data[..., t_l[mode0]:b_r[mode0], :])
            mode0, mode1 = self.matrix_mode[i]
            self.density_plane[i] = torch.nn.Parameter(self.density_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]])
            self.appearance_plane[i] = torch.nn.Parameter(self.appearance_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]])
            if self.semantic_plane is not None:
                self.semantic_plane[i] = torch.nn.Parameter(self.semantic_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]])
            if self.instance_plane is not None:
                self.instance_plane[i] = torch.nn.Parameter(self.instance_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]])
            if self.use_distilled_features_semantic or self.use_distilled_features_instance:
                self.feature_line[i] = torch.nn.Parameter(self.feature_line[i].data[..., t_l[mode0]:b_r[mode0], :])
                self.feature_plane[i] = torch.nn.Parameter(self.feature_plane[i].data[..., t_l[mode1]:b_r[mode1], t_l[mode0]:b_r[mode0]])

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.appearance_plane, self.appearance_line = self.upsample_plane_line(self.appearance_plane, self.appearance_line, res_target)
        self.density_plane, self.density_line = self.upsample_plane_line(self.density_plane, self.density_line, res_target)
        if self.semantic_plane is not None:
            self.semantic_plane, self.semantic_line = self.upsample_plane_line(self.semantic_plane, self.semantic_line, res_target)
        if self.instance_plane is not None:
            self.instance_plane, self.instance_line = self.upsample_plane_line(self.instance_plane, self.instance_line, res_target)
        if self.use_distilled_features_semantic or self.use_distilled_features_instance:
            self.feature_plane, self.feature_line = self.upsample_plane_line(self.feature_plane, self.feature_line, res_target)

    @torch.no_grad()
    def upsample_plane_line(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vector_mode)):
            vec_id = self.vector_mode[i]
            mat_id_0, mat_id_1 = self.matrix_mode[i]
            plane_coef[i] = torch.nn.Parameter(F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear', align_corners=True))
            line_coef[i] = torch.nn.Parameter(F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
        return plane_coef, line_coef

    def get_optimizable_parameters(self, lr_grid, lr_net, weight_decay=0):
        grad_vars = [{'params': self.density_line, 'lr': lr_grid, 'weight_decay': weight_decay}, {'params': self.appearance_line, 'lr': lr_grid},
                     {'params': self.density_plane, 'lr': lr_grid, 'weight_decay': weight_decay}, {'params': self.appearance_plane, 'lr': lr_grid},
                     {'params': self.appearance_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_appearance_mlp.parameters(), 'lr': lr_net}]
        if self.semantic_plane is not None:
            grad_vars.extend([
                {'params': self.semantic_plane, 'lr': lr_grid}, {'params': self.semantic_line, 'lr': lr_grid},
                {'params': self.semantic_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_semantic_mlp.parameters(), 'lr': lr_net}])
        elif self.render_semantic_mlp is not None:
            grad_vars.extend([{'params': self.render_semantic_mlp.parameters(), 'lr': lr_net}])
        if self.use_distilled_features_semantic or self.use_distilled_features_instance:
            grad_vars.extend([
                {'params': self.feature_plane, 'lr': lr_grid}, {'params': self.feature_line, 'lr': lr_grid},
                {'params': self.feature_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_feature_mlp.parameters(), 'lr': lr_net}])
        return grad_vars

    def get_optimizable_density_parameters(self, lr_grid):
        grad_vars = [{'params': self.density_line, 'lr': lr_grid}, {'params': self.density_plane, 'lr': lr_grid}]
        return grad_vars

    def get_optimizable_segment_parameters(self, lr_grid, lr_net, _weight_decay=0):
        grad_vars = []
        if self.semantic_plane is not None:
            grad_vars.extend([
                {'params': self.semantic_plane, 'lr': lr_grid}, {'params': self.semantic_line, 'lr': lr_grid},
                {'params': self.semantic_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_semantic_mlp.parameters(), 'lr': lr_net}])
        elif self.render_semantic_mlp is not None:
            grad_vars.extend([{'params': self.render_semantic_mlp.parameters(), 'lr': lr_net}])
        return grad_vars

    def get_optimizable_instance_parameters(self, lr_grid, lr_net, using_DINO=False):
        # if using_DINO is True, don't optimize slow_mlp parameters
        grad_vars = []
        if self.instance_plane is not None:
            grad_vars.extend([
                {'params': self.instance_plane, 'lr': lr_grid}, {'params': self.instance_line, 'lr': lr_grid},
                {'params': self.instance_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_instance_mlp.mlp.parameters(), 'lr': lr_net}])
        elif self.render_instance_mlp is not None:
            grad_vars.extend([{'params': self.render_instance_mlp.mlp.parameters(), 'lr': lr_net}])
            if self.use_proj:
                grad_vars.extend([{'params': self.proj_layer.fast_proj.parameters(), 'lr': lr_net}])
        
        if self.slow_fast_mode and not using_DINO: # optimize slow_mlp parameters only when not using DINO style training
            grad_vars.extend([{'params': self.render_instance_mlp.slow_mlp.parameters(), 'lr': lr_net}])
            if self.use_proj:
                grad_vars.extend([{'params': self.proj_layer.slow_proj.parameters(), 'lr': lr_net}])
        
        return grad_vars

    def tv_loss_density(self, regularizer):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + regularizer(self.density_plane[idx]) * 1e-2  # + regularizer(self.density_line[idx]) * 1e-3
        return total

    def tv_loss_appearance(self, regularizer):
        total = 0
        for idx in range(len(self.appearance_plane)):
            total = total + regularizer(self.appearance_plane[idx]) * 1e-2  # + regularizer(self.appearance_line[idx]) * 1e-3
        return total

    def tv_loss_semantics(self, regularizer):
        total = 0
        if self.semantic_plane is not None:
            for idx in range(len(self.semantic_plane)):
                total = total + regularizer(self.semantic_plane[idx]) * 1e-2 + regularizer(self.semantic_line[idx]) * 1e-3
        return total
    
    def tv_loss_instances(self, regularizer):
        total = 0
        if self.instance_plane is not None:
            for idx in range(len(self.instance_plane)):
                total = total + regularizer(self.instance_plane[idx]) * 1e-2 + regularizer(self.instance_line[idx]) * 1e-3
        return total
    
    def tv_loss_distilled_features(self, regularizer):
        total = 0
        if self.use_distilled_features_semantic or self.use_distilled_features_instance:
            for idx in range(len(self.feature_plane)):
                total = total + regularizer(self.feature_plane[idx]) * 1e-2 + regularizer(self.feature_line[idx]) * 1e-3
        return total

    def total_tv_loss(self, regularizer, config, current_epoch):
        loss_tv_density = self.tv_loss_density(regularizer)
        loss_tv_semantics = self.tv_loss_semantics(regularizer) if current_epoch >= config.late_semantic_optimization else torch.zeros_like(loss_tv_density, device=loss_tv_density.device, requires_grad=True)
        loss_tv_instances = self.tv_loss_instances(regularizer) if current_epoch >= config.instance_optimization_epoch else torch.zeros_like(loss_tv_density, device=loss_tv_density.device, requires_grad=True)
        loss_tv_appearance = self.tv_loss_appearance(regularizer)
        loss_tv = loss_tv_density * config.lambda_tv_density + \
                loss_tv_appearance * config.lambda_tv_appearance + \
                loss_tv_semantics * config.lambda_tv_semantics + \
                loss_tv_instances * config.lambda_tv_instances
        return loss_tv

    def load_weights_debug(self, weights):
        self.density_plane.load_state_dict(get_parameters_from_state_dict(weights, 'density_plane'))
        self.density_line.load_state_dict(get_parameters_from_state_dict(weights, 'density_line'))
        self.appearance_plane.load_state_dict(get_parameters_from_state_dict(weights, 'appearance_plane'))
        self.appearance_line.load_state_dict(get_parameters_from_state_dict(weights, 'appearance_line'))
        self.appearance_basis_mat.load_state_dict(get_parameters_from_state_dict(weights, 'appearance_basis_mat'))
        self.render_appearance_mlp.load_state_dict(get_parameters_from_state_dict(weights, 'render_appearance_mlp'))
        if self.num_semantics_comps is not None:
            if self.semantic_plane is not None:
                self.semantic_plane.load_state_dict(get_parameters_from_state_dict(weights, 'semantic_plane'))
                self.semantic_line.load_state_dict(get_parameters_from_state_dict(weights, 'semantic_line'))
                self.semantic_basis_mat.load_state_dict(get_parameters_from_state_dict(weights, 'semantic_basis_mat'))
            self.render_semantic_mlp.load_state_dict(get_parameters_from_state_dict(weights, 'render_semantic_mlp'))
        if self.dim_feature_instance is not None:
            if self.instance_plane is not None:
                self.instance_plane.load_state_dict(get_parameters_from_state_dict(weights, 'instance_plane'))
                self.instance_line.load_state_dict(get_parameters_from_state_dict(weights, 'instance_line'))
                self.instance_basis_mat.load_state_dict(get_parameters_from_state_dict(weights, 'instance_basis_mat'))
            self.render_instance_mlp.load_state_dict(get_parameters_from_state_dict(weights, 'render_instance_mlp'))
        if self.use_distilled_features_semantic or self.use_distilled_features_instance:
            self.feature_plane.load_state_dict(get_parameters_from_state_dict(weights, 'feature_plane'))
            self.feature_line.load_state_dict(get_parameters_from_state_dict(weights, 'feature_line'))
            self.feature_basis_mat.load_state_dict(get_parameters_from_state_dict(weights, 'feature_basis_mat'))
            self.render_feature_mlp.load_state_dict(get_parameters_from_state_dict(weights, 'render_feature_mlp'))



class ConditionalTensorVMSplit(TensorVMSplit):
    '''
    Variant of TensorVMSplit that is conditional on a latent code.
    Notes:
        - all brances are still separate or parallel
        - the latent code is used to condition all branches
        - density branch uses a MLP to accommodate the latent code
    '''
    def __init__(self, grid_dim, latent_dim=16,
                 num_density_comps=(16, 16, 16), num_appearance_comps=(48, 48, 48), num_semantics_comps=None, num_instance_comps=None,
                 dim_density=12, dim_appearance=27, dim_semantics=27, dim_instances=27, splus_density_shift=-10, pe_view=2, pe_feat=2, 
                 dim_mlp_density=32, dim_mlp_color=128, dim_mlp_semantics=128, dim_mlp_instance=256, 
                 num_semantic_classes=0, dim_feature_instance=None,
                 output_mlp_semantics=torch.nn.Softmax(dim=-1),
                 use_semantic_mlp=False, use_instance_mlp=False, use_feature_reg=False,
                 use_distilled_features_semantic=False, use_distilled_features_instance=False, num_feature_comps=(48, 48, 48),
                 pe_sem=0, pe_ins=0,
                 slow_fast_mode=False, use_proj=False):
        super(ConditionalTensorVMSplit, self).__init__(
            grid_dim=grid_dim, num_density_comps=num_density_comps, num_appearance_comps=num_appearance_comps, num_semantics_comps=num_semantics_comps, num_instance_comps=num_instance_comps,
            dim_appearance=dim_appearance, dim_semantics=dim_semantics, dim_instances=dim_instances, splus_density_shift=splus_density_shift, pe_view=pe_view, pe_feat=pe_feat,
            dim_mlp_color=dim_mlp_color, dim_mlp_semantics=dim_mlp_semantics, dim_mlp_instance=dim_mlp_instance,
            num_semantic_classes=num_semantic_classes, dim_feature_instance=dim_feature_instance,
            output_mlp_semantics=output_mlp_semantics,
            use_semantic_mlp=use_semantic_mlp, use_instance_mlp=use_instance_mlp, use_feature_reg=use_feature_reg,
            use_distilled_features_semantic=use_distilled_features_semantic, use_distilled_features_instance=use_distilled_features_instance, num_feature_comps=num_feature_comps,
            pe_sem=pe_sem, pe_ins=pe_ins,
            slow_fast_mode=slow_fast_mode, use_proj=use_proj)
        
        self.dim_density = dim_density
        self.dim_mlp_density = dim_mlp_density
        self.latent_dim = latent_dim
        self.density_basis_mat = torch.nn.Linear(sum(self.num_density_comps), self.dim_density, bias=False)
        self.render_density_mlp = ConditionalMLPRenderFeature(
            in_channels=dim_density, latent_channels=latent_dim,
            out_channels=1, pe_view=0, pe_feat=0, dim_mlp=dim_mlp_density, 
            output_activation=torch.nn.Softplus(), splus_density_shift=self.splus_density_shift)
        self.render_appearance_mlp = ConditionalMLPRenderFeature(
            in_channels=dim_appearance, latent_channels=latent_dim,
            out_channels=3, pe_view=pe_view, pe_feat=pe_feat, dim_mlp=dim_mlp_color)
        # TODO: add support for semantic and instance branches

    def compute_density(self, xyz_sampled, latents):
        density_feature = self.compute_feature(xyz_sampled, self.density_plane, self.density_line, self.density_basis_mat)
        return self.render_density_mlp(None, density_feature, latents)

    def get_optimizable_parameters(self, lr_grid, lr_net, weight_decay=0):
        grad_vars = [{'params': self.density_line, 'lr': lr_grid, 'weight_decay': weight_decay}, {'params': self.appearance_line, 'lr': lr_grid},
                     {'params': self.density_plane, 'lr': lr_grid, 'weight_decay': weight_decay}, {'params': self.appearance_plane, 'lr': lr_grid},
                     {'params': self.density_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_density_mlp.parameters(), 'lr': lr_net},
                     {'params': self.appearance_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_appearance_mlp.parameters(), 'lr': lr_net}]
        if self.semantic_plane is not None:
            grad_vars.extend([
                {'params': self.semantic_plane, 'lr': lr_grid}, {'params': self.semantic_line, 'lr': lr_grid},
                {'params': self.semantic_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_semantic_mlp.parameters(), 'lr': lr_net}])
        elif self.render_semantic_mlp is not None:
            grad_vars.extend([{'params': self.render_semantic_mlp.parameters(), 'lr': lr_net}])
        if self.use_distilled_features_semantic or self.use_distilled_features_instance:
            grad_vars.extend([
                {'params': self.feature_plane, 'lr': lr_grid}, {'params': self.feature_line, 'lr': lr_grid},
                {'params': self.feature_basis_mat.parameters(), 'lr': lr_net}, {'params': self.render_feature_mlp.parameters(), 'lr': lr_net}])
        return grad_vars


class MLPRenderFeature(torch.nn.Module):

    def __init__(self, in_channels, out_channels=3, pe_view=2, pe_feat=2, dim_mlp_color=128, output_activation=torch.sigmoid):
        super().__init__()
        self.pe_view = pe_view
        self.pe_feat = pe_feat
        self.output_channels = out_channels
        self.view_independent = self.pe_view == 0 and self.pe_feat == 0
        self.in_feat_mlp = 2 * pe_view * 3 + 2 * pe_feat * in_channels + in_channels + (3 if not self.view_independent else 0)
        self.output_activation = output_activation
        layer1 = torch.nn.Linear(self.in_feat_mlp, dim_mlp_color)
        layer2 = torch.nn.Linear(dim_mlp_color, dim_mlp_color)
        layer3 = torch.nn.Linear(dim_mlp_color, out_channels)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, viewdirs, features):
        indata = [features]
        if not self.view_independent:
            indata.append(viewdirs)
        if self.pe_feat > 0:
            indata += [MLPRenderFeature.positional_encoding(features, self.pe_feat)]
        if self.pe_view > 0:
            indata += [MLPRenderFeature.positional_encoding(viewdirs, self.pe_view)]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        out = self.output_activation(out)
        return out

    @staticmethod
    def positional_encoding(positions, freqs):
        freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
        pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] + (freqs * positions.shape[-1],))
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts
    

class ConditionalMLPRenderFeature(torch.nn.Module):

    def __init__(self, in_channels, latent_channels, 
                 out_channels=3, pe_view=2, pe_feat=2, dim_mlp=128, output_activation=torch.sigmoid,
                 splus_density_shift=None):
        super().__init__()
        if not isinstance(output_activation, torch.nn.Softplus):
            assert splus_density_shift is None, 'splus_density_shift is only used when output_activation is Softplus'
        self.latent_channels = latent_channels
        self.pe_view = pe_view
        self.pe_feat = pe_feat
        self.output_channels = out_channels
        self.splus_density_shift = splus_density_shift
        self.view_independent = self.pe_view == 0 and self.pe_feat == 0
        self.in_feat_mlp = 2 * pe_view * 3 + 2 * pe_feat * in_channels + in_channels + (3 if not self.view_independent else 0)
        self.in_feat_mlp += latent_channels
        self.output_activation = output_activation
        layer1 = torch.nn.Linear(self.in_feat_mlp, dim_mlp)
        layer2 = torch.nn.Linear(dim_mlp, dim_mlp)
        layer3 = torch.nn.Linear(dim_mlp, out_channels, bias=(not isinstance(output_activation, torch.nn.Softplus))) # no bias for Softplus

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        if self.mlp[-1].bias is not None: torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, viewdirs, features, latents):
        indata = [features]
        if not self.view_independent:
            indata.append(viewdirs)
        if self.pe_feat > 0:
            indata += [MLPRenderFeature.positional_encoding(features, self.pe_feat)]
        if self.pe_view > 0:
            indata += [MLPRenderFeature.positional_encoding(viewdirs, self.pe_view)]
        indata += [latents]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        if isinstance(self.output_activation, torch.nn.Softplus):
            out += self.splus_density_shift
        out = self.output_activation(out)
        return out


class MLPRenderInstanceFeature(torch.nn.Module):

    def __init__(self, in_channels, out_channels, num_mlp_layers=5, dim_mlp=256, pe_feat=0, output_activation=torch.nn.Softmax(dim=-1), use_features=False,
                 slow_fast_mode=False):
        super().__init__()
        self.output_channels = out_channels
        self.output_activation = output_activation
        self.pe_feat = pe_feat
        self.use_features = use_features
        self.slow_fast_mode = slow_fast_mode # if true, two MLPs are used, one for slow and one for fast features 
        self.in_feat_mlp = 2 * pe_feat * in_channels + in_channels
        if use_features:
            self.in_feat_mlp += 64
        layers = [torch.nn.Linear(self.in_feat_mlp, dim_mlp)]
        for i in range(num_mlp_layers - 2):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(dim_mlp, dim_mlp))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(dim_mlp, out_channels))
        self.mlp = torch.nn.Sequential(*layers)

        if self.slow_fast_mode:
            # create slow_mlp with same architecture (but NOT the same weights) as mlp
            slow_layers = [torch.nn.Linear(self.in_feat_mlp, dim_mlp)]
            for i in range(num_mlp_layers - 2):
                slow_layers.append(torch.nn.ReLU(inplace=True))
                slow_layers.append(torch.nn.Linear(dim_mlp, dim_mlp))
            slow_layers.append(torch.nn.ReLU(inplace=True))
            slow_layers.append(torch.nn.Linear(dim_mlp, out_channels))
            self.slow_mlp = torch.nn.Sequential(*slow_layers)

        # NOTE: Commented out for DINO experiment
        # torch.nn.init.constant_(self.mlp[-1].bias, 0)
        # torch.nn.init.constant_(self.slow_mlp[-1].bias, 0)

    def forward(self, distilled_feats, feat_xyz):
        indata = [feat_xyz]
        if self.pe_feat > 0:
            indata += [MLPRenderFeature.positional_encoding(feat_xyz, self.pe_feat)]
        if self.use_features:
            assert distilled_feats is not None, 'Distilled features are required for this model'
            indata += [distilled_feats]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        out = self.output_activation(out)
        if self.slow_fast_mode:
            slow_out = self.slow_mlp(mlp_in)
            slow_out = self.output_activation(slow_out)
            out = torch.cat([out, slow_out], dim=-1) # concat slow and fast features
        return out
    

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class SlowFastProjLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.fast_proj = torch.nn.Linear(in_channels, out_channels)
        # self.slow_proj = torch.nn.Linear(in_channels, out_channels)
        self.fast_proj = DINOHead(in_channels, out_channels, nlayers=1, bottleneck_dim=8)
        self.slow_proj = DINOHead(in_channels, out_channels, nlayers=1, bottleneck_dim=8)

    def forward(self, fast_x, slow_x):
        fast_out = self.fast_proj(fast_x)
        slow_out = self.slow_proj(slow_x)
        return fast_out, slow_out
    

class MLPRenderSemanticFeature(torch.nn.Module):

    def __init__(self, in_channels, out_channels, pe_feat=0, num_mlp_layers=5, dim_mlp=256, output_activation=torch.nn.Identity(), use_features=False):
        super().__init__()
        self.output_channels = out_channels
        self.output_activation = output_activation
        self.pe_feat = pe_feat
        self.use_features = use_features
        self.in_feat_mlp = 2 * pe_feat * in_channels + in_channels
        if use_features:
            self.in_feat_mlp += 64
        layers = [torch.nn.Linear(self.in_feat_mlp, dim_mlp)]
        for i in range(num_mlp_layers - 2):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(dim_mlp, dim_mlp))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(dim_mlp, out_channels))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, distilled_feats, feat_xyz):
        indata = [feat_xyz]
        if self.pe_feat > 0:
            indata += [MLPRenderFeature.positional_encoding(feat_xyz, self.pe_feat)]
        if self.use_features:
            assert distilled_feats is not None, 'Distilled features are required for this model'
            indata += [distilled_feats]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        out = self.output_activation(out)
        return out


class MLPRenderSemanticFeatureWithRegularization(torch.nn.Module):

    def __init__(self, in_channels, out_channels, pe_feat=0, num_mlp_layers=5, dim_mlp=256, output_activation=torch.nn.Identity()):
        super().__init__()
        self.output_channels = out_channels
        self.output_activation = output_activation
        self.pe_feat = pe_feat
        self.in_feat_mlp = 2 * pe_feat * in_channels + in_channels
        layers = [torch.nn.Linear(self.in_feat_mlp, dim_mlp)]
        for i in range(num_mlp_layers - 3):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(dim_mlp, dim_mlp))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(dim_mlp, 384))
        self.mlp_backbone = torch.nn.Sequential(*layers)
        self.head_class = torch.nn.Linear(384, out_channels)

    def forward(self, _dummy, feat_xyz):
        out = self.get_backbone_feats(feat_xyz)
        out = self.head_class(out)
        out = self.output_activation(out)
        return out

    def get_backbone_feats(self, feat_xyz):
        indata = [feat_xyz]
        if self.pe_feat > 0:
            indata += [MLPRenderFeature.positional_encoding(feat_xyz, self.pe_feat)]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp_backbone(mlp_in)
        return out


def render_features_direct(_viewdirs, appearance_features):
    return appearance_features


def render_features_direct_with_softmax(_viewdirs, appearance_features):
    return torch.nn.Softmax(dim=-1)(appearance_features)
