# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from attrdict import AttrDict

"""Model graph definitions and other functions for training and testing."""

import functools
import math
import operator
import os
import random
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(config, gpuid):
    """Make model instance and pin to one gpu.

  Args:
    config: arguments.
    gpuid: gpu id to use
  Returns:
    Model instance.
  """

    model = Model(config, '%s' % config.modelname).to(torch.device(f"cuda:{gpuid}"))
    return model


class Conv2d(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel, padding='SAME', stride=1,
                 activation=torch.nn.Identity, add_bias=True, data_format='NHWC',
                 w_init=None, scope='conv'):
        super(Conv2d, self).__init__(
            in_channel, out_channel, kernel, stride=stride, bias=add_bias
        )
        self.scope = scope
        self.activation = activation()
        self.data_format = data_format
        nn.init.kaiming_normal_(self.weight)
        if add_bias:
            nn.init.constant_(self.bias,0)

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i) - 1) + (k - 1) * d + 1 - i, 0)

    def forward(self, x):

        if self.data_format == "NHWC":
            x = x.permute(0, 3, 1, 2)

        N, C, H, W = x.shape
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        x = self.activation(x)
        if self.data_format == "NHWC":
            x = x.permute(0, 2, 3, 1)
        return x


class Linear(nn.Module):
    def __init__(self, input_size, output_size, scope="", add_bias=True, activation=nn.Identity):
        super(Linear, self).__init__()
        self.linear = nn.Linear(
            input_size,
            output_size,
            bias=add_bias
        )
        self.scope = scope
        self.activation = activation()
        nn.init.trunc_normal_(self.linear.weight, std=0.1)
        if add_bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class LSTMStateTuple:
    def __init__(self, h, c):
        self.h = h
        self.c = c


class LSTM(nn.Module):
    def __init__(self, scope="", *args, **kwargs):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(*args, **kwargs, batch_first=True)
        self.scope = scope

    def forward(self, x, state=None):
        if state is not None:
            h, c = state
            output, (h, c) = self.lstm(x, (h, c))
        else:
            output, (h, c) = self.lstm(x)
        return output, LSTMStateTuple(h[0], c[0])


class LSTMCell(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LSTMCell, self).__init__()
        self.lstm = nn.LSTMCell(*args, **kwargs)

    def forward(self, x, state=None):
        if state is not None:
            h, c = state.h, state.c
            (h, c) = self.lstm(x, (h, c))
            return LSTMStateTuple(h, c)
        else:
            (h, c) = self.lstm(x)
            return LSTMStateTuple(h, c)

def gather_nd(params, indices):
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:]

    """
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]    # (num_samples, ...)
    return output.reshape(out_shape).contiguous()

def cond(condition, fn1, fn2):
    if condition:
        return fn1()
    else:
        return fn2()


class Model(nn.Module):
    """Model graph definitions.
  """

    def __init__(self, config, scope):
        super(Model, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scope = scope

        # self.global_step = tf.get_variable('global_step', shape=[],
        #                                    dtype=torch.int32,
        #                                    initializer=tf.constant_initializer(0),
        #                                    trainable=False)

        # get all the dimension here
        # Tensor dimensions, so pylint: disable=g-bad-name
        N = self.N = config.batch_size

        KP = self.KP = config.kp_size

        SH = self.SH = config.scene_h
        SW = self.SW = config.scene_w
        SC = self.SC = config.scene_class

        K = self.K = config.max_other

        self.P = P = 2  # traj coordinate dimension

        self.enc_traj = LSTM(
            input_size=config.emb_size,
            hidden_size=config.enc_hidden_size,
            dropout=config.keep_prob
        )

        self.enc_personscene = LSTM(
            input_size=config.scene_conv_dim,
            hidden_size=config.enc_hidden_size,
            dropout=config.keep_prob
        )
        if config.add_kp:
            self.enc_kp = LSTM(
                input_size=config.emb_size,
                hidden_size=config.enc_hidden_size,
                dropout=config.keep_prob
            )

        self.enc_person = LSTM(
            input_size=config.person_feat_dim,
            hidden_size=config.enc_hidden_size,
            dropout=config.keep_prob
        )
        self.enc_other = LSTM(
            input_size=config.box_emb_size * 2,
            hidden_size=config.enc_hidden_size,
            dropout=config.keep_prob
        )

        self.enc_gridclass = nn.ModuleDict()
        for i, (h,w) in enumerate(config.scene_grids):
            self.enc_gridclass['enc_gridclass_%s' % i] = LSTM(
                input_size=h*w,
                hidden_size=config.enc_hidden_size,
                dropout=config.keep_prob
            )
        # ------------------------ decoder

        if config.multi_decoder:
            self.dec_cell_traj = nn.ModuleDict()
            for i in range(len(config.traj_cats)):
                self.dec_cell_traj[str(i)] = LSTMCell(
                    input_size=config.emb_size + config.enc_hidden_size,
                    hidden_size=config.dec_hidden_size,
                )
        else:
            self.dec_cell_traj = LSTMCell(
                input_size=config.emb_size + config.enc_hidden_size,
                hidden_size=config.dec_hidden_size,
            )

        self.enc_xy_emb = Linear(2, output_size=config.emb_size,
                                 activation=config.activation_func,
                                 add_bias=True, scope="enc_xy_emb")
        conv_dim = config.scene_conv_dim
        self.conv2 = Conv2d(
            in_channel=config.scene_class,
            out_channel=conv_dim,
            kernel=config.scene_conv_kernel,
            stride=2, activation=config.activation_func,
            add_bias=True, scope='conv2'
        )

        self.conv3 = Conv2d(
            in_channel=conv_dim,
            out_channel=conv_dim,
            kernel=config.scene_conv_kernel,
            stride=2, activation=config.activation_func,
            add_bias=True, scope='conv3')

        self.kp_emb = Linear(input_size=config.kp_size * 2, output_size=config.emb_size, add_bias=True,
                             activation=config.activation_func)

        self.other_box_geo_emb = Linear(
            input_size=4,
            add_bias=True,
            activation=config.activation_func, output_size=config.box_emb_size,
            scope='other_box_geo_emb')

        self.other_box_class_emb = Linear(
            input_size=config.scene_class -1 ,
            add_bias=True,
            activation=config.activation_func, output_size=config.box_emb_size,
            scope='other_box_class_emb')
        self.traj_label_emb = Linear(
            input_size=config.num_act,
            output_size=config.enc_hidden_size, add_bias=True,
            activation=config.activation_func,
            scope="embed_traj_label"
        )

        self.obs_grid_class_emb = nn.ModuleDict()
        self.grid_class_conv = nn.ModuleDict()
        self.grid_target_conv = nn.ModuleDict()
        for i, (h, w)  in enumerate(config.scene_grids):
            self.obs_grid_class_emb[str(i)] = Linear(
                input_size=1,
                output_size=config.emb_size,
                activation=config.activation_func,
                add_bias=True, scope=f'obs_grid_class_emb_{i}')
            feature_size = config.enc_hidden_size * 7 + (config.enc_hidden_size if config.embed_traj_label else 0)

            scene_feature_size = config.emb_size + \
                                 conv_dim + \
                                 feature_size  # h_tile + conv + xyembed
            self.grid_class_conv[str(i)] = Conv2d(
                in_channel=scene_feature_size,
                out_channel=1, kernel=1, stride=1,
                activation=config.activation_func,
                add_bias=True, scope=f'grid_class_{i}'
            )
            self.grid_target_conv[str(i)] = Conv2d(
                in_channel=scene_feature_size,
                out_channel=2, kernel=1, stride=1,
                activation=config.activation_func,
                add_bias=True,
                scope=f'grid_target_{i}'
            )

        self.traj_cat_logits = Linear(
            input_size=feature_size,
            output_size=len(config.traj_cats),
            add_bias=False,
            scope='traj_cat_logits')
        self.future_act = Linear(
            input_size=feature_size,
            output_size=config.num_act, add_bias=False, scope='future_act')

        self.out_xy_mlp2 = Linear(
            input_size=config.dec_hidden_size,
            output_size=P,
            add_bias=False, scope='out_xy_mlp2')

        self.xy_emb_dec = Linear(
            input_size=2,
            output_size=config.emb_size,
            activation=config.activation_func, add_bias=True,
            scope='xy_emb_dec'
        )

    def forward(self, batch):
        config = self.config
        out = AttrDict()

        N = self.N
        KP = self.KP

        obs_length = batch['traj_obs_gt_mask'].shape[1]
        traj_xy_emb_enc = self.enc_xy_emb(batch['traj_obs_gt'])
        traj_obs_enc_h, traj_obs_enc_last_state = self.enc_traj(
            traj_xy_emb_enc)  # encode traj
        enc_h_list = [traj_obs_enc_h]
        enc_last_state_list = [traj_obs_enc_last_state]
        # grid class and grid regression encoder
        # multi-scale
        grid_obs_enc_h = []
        grid_obs_enc_last_state = []

        for i, (h, w) in enumerate(config.scene_grids):
            #  [N, T] -> [N, T, h*w]
            obs_gridclass_onehot = F.one_hot(batch['grid_obs_labels'][i].long(), h * w).float()
            obs_gridclass_encode_h, obs_gridclass_encode_last_state = \
                self.enc_gridclass['enc_gridclass_%s' % i](obs_gridclass_onehot)
            grid_obs_enc_h.append(obs_gridclass_encode_h)
            grid_obs_enc_last_state.append(obs_gridclass_encode_last_state)
        enc_h_list.extend(grid_obs_enc_h)
        enc_last_state_list.extend(grid_obs_enc_last_state)

        #########################Scope scene#################################3
        """ embedding_lookup """
        obs_scene = torch.index_select(batch['scene_feat'], 0, batch['obs_scene'].long().ravel())
        obs_scene = obs_scene.reshape(N, -1, *batch['scene_feat'].shape[1:])
        obs_scene = torch.mean(obs_scene, dim=1)  # [N, SH, SW, SC]

        """scene conv"""
        scene_conv1 = obs_scene
        scene_conv2 = self.conv2(scene_conv1)
        scene_conv3 = self.conv3(scene_conv2)
        scene_convs = [scene_conv2, scene_conv3]

        # pool the scene features for each trajectory, for different scale
        # currently only used single scale conv
        pool_scale_idx = config.pool_scale_idx
        scene_h, scene_w = config.scene_grids[pool_scale_idx]
        conv_dim = config.scene_conv_dim

        # [N, num_grid_class, conv_dim]
        scene_conv_full = scene_convs[pool_scale_idx].reshape(N, scene_h * scene_w, conv_dim)

        # [N, seq_len]
        obs_grid = batch['grid_obs_labels'][pool_scale_idx]

        obs_grid = obs_grid.ravel()  # [N*seq_len]
        # [N*seq_len, 2]
        indices = torch.stack(
            [torch.arange(obs_grid.shape[0]).to(obs_grid), obs_grid], dim=-1).long()

        # [N, seq_len, num_grid_class, conv_dim]
        scene_conv_full_tile = torch.unsqueeze(
            scene_conv_full, 1).repeat(1, config.obs_len, 1, 1)
        # [N*seq_len, num_grid_class, conv_dim]
        scene_conv_full_tile = \
            scene_conv_full_tile.reshape(-1, scene_h * scene_w, conv_dim)

        # [N*seq_len, h*w, feat_dim] + [N*seq_len,2] -> # [N*seq_len, feat_dim]
        obs_personscene = gather_nd(scene_conv_full_tile, indices)
        obs_personscene = obs_personscene.reshape(N, config.obs_len, conv_dim)

        # obs_personscene [N, seq_len, conv_dim]
        personscene_obs_enc_h, personscene_obs_enc_last_state = \
            self.enc_personscene(obs_personscene)
        enc_h_list.append(personscene_obs_enc_h)
        enc_last_state_list.append(personscene_obs_enc_last_state)

        # person pose
        if config.add_kp:
            obs_kp = batch['obs_kp'].reshape(N, -1, KP * 2)
            obs_kp = self.kp_emb(obs_kp)
            # obs_kp = tf.Print(obs_kp, [obs_kp], "obs_kp")
            kp_obs_enc_h, kp_obs_enc_last_state = self.enc_kp(obs_kp)

            enc_h_list.append(kp_obs_enc_h)
            enc_last_state_list.append(kp_obs_enc_last_state)

        # person appearance
        # average and then normal lstm
        obs_person_features = batch['obs_person_features'].mean(dim=[2, 3])
        # [N,T,hdim]
        person_obs_enc_h, person_obs_enc_last_state = self.enc_person(obs_person_features)
        enc_h_list.append(person_obs_enc_h)
        enc_last_state_list.append(person_obs_enc_last_state)

        # extract features from other boxes
        # obs_other_boxes [N, obs_len, K, 4]
        # obs_other_boxes_class [N, obs_len, K, num_class]
        # obs_other_boxes_mask [N, obs_len, K]

        """other_box"""
        obs_other_boxes_geo_features = self.other_box_geo_emb(batch['obs_other_boxes'])
        obs_other_boxes_class_features = self.other_box_class_emb(batch['obs_other_boxes_class'])
        obs_other_boxes_features = torch.cat(
            [obs_other_boxes_geo_features, obs_other_boxes_class_features],
            dim=3)

        # cosine simi
        obs_other_boxes_geo_features = F.normalize(obs_other_boxes_geo_features, p=2, dim=-1)
        obs_other_boxes_class_features = F.normalize(
            obs_other_boxes_class_features, p=2, dim=-1)
        # [N, T,K]
        other_attention = torch.multiply(
            obs_other_boxes_geo_features, obs_other_boxes_class_features).sum(3)

        other_attention = exp_mask(
            other_attention, batch['obs_other_boxes_mask'])

        other_attention = torch.softmax(other_attention,dim=-1)
        # [N, obs_len, K, 1] * [N, obs_len, K, feat_dim]
        # -> [N, obs_len, feat_dim]
        other_box_features_attended = (torch.unsqueeze(
            other_attention, -1) * obs_other_boxes_features).mean(dim=2)

        other_obs_enc_h, other_obs_enc_last_state = self.enc_other(other_box_features_attended)
        enc_h_list.append(other_obs_enc_h)
        enc_last_state_list.append(other_obs_enc_last_state)

        # pack all observed hidden states
        obs_enc_h = torch.stack(enc_h_list, dim=1)
        # .h is [N,h_dim*k]
        obs_enc_last_state = concat_states(enc_last_state_list, dim=1)

        if config.add_activity and config.embed_traj_label:
            batch['future_act_label'].to("float32")
            label_feat = self.traj_label_emb(
                batch['future_act_label'].float()
            )

            new_h = torch.cat([label_feat, obs_enc_last_state.h], -1)
            # new_h = tf.Print(new_h, [new_h], "new_h")
            obs_enc_last_state = LSTMStateTuple(h=new_h, c=obs_enc_last_state.c)

        # -------------------------------------------------- xy decoder
        traj_obs_last = batch['traj_obs_gt'][:, -1]

        pred_length = batch['traj_pred_gt_mask'].long().sum(1)  # N
        if config.multi_decoder:
            # [N, num_traj_cat] # each is num_traj_cat classification
            traj_class_logits = self.traj_class_head(
                obs_enc_h, obs_enc_last_state, scope='traj_class_predict')
            out['traj_class_logits'] = traj_class_logits
            # [N]
            traj_class = torch.argmax(traj_class_logits, dim=1)
            traj_class_gated = cond(
                self.training,
                lambda: batch['traj_class_gt'],
                lambda: traj_class,
            )
            traj_pred_outs = [
                self.decoder(
                    batch,
                    traj_obs_last,
                    traj_obs_enc_last_state,
                    obs_enc_h,
                    pred_length,
                    self.dec_cell_traj[str(traj_cat)])
                for _, traj_cat in config.traj_cats
            ]
            traj_pred_outs = torch.stack(traj_pred_outs, dim=1)

            # [N, 2]
            indices = torch.stack(
                [torch.arange(N).to(traj_class_gated), traj_class_gated.long()], dim=1)

            # [N, T, 2]
            traj_pred_out = gather_nd(traj_pred_outs, indices)

        else:
            traj_pred_out = self.decoder(traj_obs_last, traj_obs_enc_last_state,
                                         obs_enc_h, pred_length, self.dec_cell_traj, scope='decoder')
        out['traj_pred_out'] = traj_pred_out

        if config.add_activity:
            # activity decoder
            future_act_logits = self.activity_head(
                obs_enc_h, obs_enc_last_state, scope='activity_predict')
            out['future_act_logits'] = future_act_logits

        # predict the activity destination
        # scope grid head
        conv_dim = config.scene_conv_dim

        assert len(config.scene_grids) == 2
        # grid class and grid target output
        out['grid_class_logits'] = []
        out['grid_target_logits'] = []

        for i, (h, w) in enumerate(config.scene_grids):
            # [h,w,c]
            this_scene_conv = scene_convs[i]
            this_scene_conv = this_scene_conv.reshape(N, h * w, conv_dim)

            # tile
            # [N, h*w, h_dim*k]
            h_tile = torch.unsqueeze(
                obs_enc_last_state.h, dim=1).repeat(1, h * w, 1)

            # [N, h*w, conv_dim + h_dim + emb]

            scene_feature = torch.cat(
                [h_tile, this_scene_conv], dim=-1)

            # add the occupation map, grid obs input is already in the h_tile
            # [N, T, h*w]
            obs_gridclass_onehot = F.one_hot(
                batch['grid_obs_labels'][i].long(), h * w)
            obs_gridclass_occupy = obs_gridclass_onehot.sum(1)
            obs_gridclass = obs_gridclass_occupy.float()  # [N,h*w]
            obs_gridclass = obs_gridclass.reshape(N, h * w, 1)

            # [N, h*w, 1] -> [N, h*w, emb]
            obs_grid_class_emb = self.obs_grid_class_emb[str(i)](obs_gridclass)

            scene_feature = torch.cat(
                [scene_feature, obs_grid_class_emb], dim=-1)

            grid_class_logit = self.grid_class_conv[str(i)](scene_feature.reshape(N, h, w, -1))
            grid_target_logit_all = self.grid_target_conv[str(i)](scene_feature.reshape(N, h, w, -1))
            grid_class_logit = \
                grid_class_logit.reshape(N, h * w, 1)
            grid_target_logit_all = \
                grid_target_logit_all.reshape(N, h * w, 2)

            grid_class_logit = torch.squeeze(grid_class_logit, dim=-1)

            # [N]
            target_class = torch.argmax(grid_class_logit, dim=-1)

            # [N,2]
            indices = torch.stack(
                [torch.arange(N).to(target_class), target_class.long()], dim=-1)
            # [N,h*w,2] + [N,2] -> # [N,2]
            grid_target_logit = gather_nd(grid_target_logit_all, indices)

            out['grid_class_logits'].append(grid_class_logit)
            out['grid_target_logits'].append(grid_target_logit)

        return out

    # output [N, num_decoder]
    # enc_h for future extension, so pylint: disable=unused-argument
    def traj_class_head(self, enc_h, enc_last_state, scope='predict_traj_cat'):
        """Trajectory classification branch."""
        config = self.config
        feature = enc_last_state.h

        # [N, num_traj_class]
        logits = self.traj_cat_logits(feature)

        return logits

    def activity_head(self, enc_h, enc_last_state, scope='activity_predict'):
        """Activity prediction branch."""
        config = self.config

        feature = enc_last_state.h
        # feature = tf.Print(feature, [feature], scope, summarize=50)

        future_act = self.future_act(feature)

        return future_act

    def decoder(self, batch, first_input, enc_last_state, enc_h, pred_length, rnn_cell):
        """Decoder definition."""
        config = self.config
        # Tensor dimensions, so pylint: disable=g-bad-name
        N = self.N
        P = self.P

        # TODO check
        # these input only used during training
        time_1st_traj_pred = batch['traj_pred_gt'].permute(1, 0, 2)  # [N,T2,W] -> [T2,N,W]
        T2 = time_1st_traj_pred.shape[0]
        traj_pred_gt = batch['traj_pred_gt'].clone()
        traj_pred_gt = traj_pred_gt.permute(1,0,2)

        curr_cell_state = enc_last_state
        decoder_out_ta = [first_input]
        for i in range(T2):
            curr_input_xy = cond(
                self.training,
                lambda: cond(
                    i > 0,
                    lambda: traj_pred_gt[i],
                    lambda: first_input
                ),
                lambda: decoder_out_ta[-1]
            )
            xy_emb = self.xy_emb_dec(curr_input_xy)
            attended_encode_states = focal_attention(
                curr_cell_state.h, enc_h, use_sigmoid=False)

            rnn_input = torch.cat(
                [xy_emb, attended_encode_states], dim=1)

            next_cell_state = rnn_cell(rnn_input, curr_cell_state)

            decoder_out_ta.append(self.hidden2xy(next_cell_state.h))
            curr_cell_state = next_cell_state

        decoder_out = torch.stack(decoder_out_ta[1:],dim=0)  # [T2,N,h_dim]
        # [N,T2,h_dim]
        decoder_out = decoder_out.permute(1, 0, 2)
        # decoder_out = self.hidden2xy(
        #     decoder_out_h)
        return decoder_out

    def hidden2xy(self, lstm_h):
        """Hiddent states to xy coordinates."""
        # Tensor dimensions, so pylint: disable=g-bad-name
        out_xy = self.out_xy_mlp2(lstm_h)
        return out_xy

    def cal_loss(self, batch, out):
        """Model loss."""
        config = self.config
        loss_dict = AttrDict()
        # N,T,W
        # L2 loss
        # [N,T2,W]
        traj_pred_out = out['traj_pred_out']

        traj_pred_gt = batch['traj_pred_gt']

        # diff = traj_pred_out - traj_pred_gt

        xyloss = F.mse_loss(traj_pred_gt, traj_pred_out)  # [N,T2,2]

        loss_dict['xyloss'] = xyloss

        # trajectory classification loss
        if config.multi_decoder:
            traj_class_loss = nn.CrossEntropyLoss()(
                target=batch['traj_class_gt'].long(), input=out['traj_class_logits']
            )
            traj_class_loss *= config.traj_class_loss_weight
            loss_dict['traj_class_loss'] = traj_class_loss

        # ------------------------ activity destination loss
        grid_loss_weight = config.grid_loss_weight
        for i, _ in enumerate(config.scene_grids):
            grid_pred_label = batch['grid_pred_labels'][i]  # [N]
            grid_pred_target = batch['grid_pred_targets'][i]  # [N,2]

            grid_class_logit = out['grid_class_logits'][i]  # [N,h*w]
            grid_target_logit = out['grid_target_logits'][i]  # [N,2]

            # classification loss

            class_loss = nn.CrossEntropyLoss()(
                target=grid_pred_label.long(), input=grid_class_logit) * grid_loss_weight

            # regression loss
            regression_loss = nn.HuberLoss()(
                target=grid_pred_target, input=grid_target_logit)

            regression_loss = regression_loss * grid_loss_weight
            loss_dict[f'grid_class_loss_{i}'] = class_loss
            loss_dict[f'grid_regression_loss_{i}'] = regression_loss

        # --------- activity class loss
        if config.add_activity:
            act_loss_weight = config.act_loss_weight
            future_act_logits = out['future_act_logits']  # [N,num_act]
            future_act_label = batch['future_act_label']  # [N,num_act]

            activity_loss = nn.BCEWithLogitsLoss()(
                target=future_act_label.float(), input=future_act_logits) * act_loss_weight
            loss_dict['activity_loss'] = activity_loss
        total_loss = sum(loss_dict.values())
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        # if config.wd is not None:
        #     wd = wd_cost('.*/W', config.wd, scope='wd_cost')
        #     if wd:
        #         wd = tf.add_n(wd)

        # there might be l2 weight loss in some layer
        # self.loss = tf.add_n(losses, name='total_losses')
        return loss_dict, total_loss

    def encode_other_boxes(self, person_box, other_box):
        """Encoder other boxes."""
        # get relative geometric feature
        x1, y1, x2, y2 = person_box
        xx1, yy1, xx2, yy2 = other_box

        x_m = x1
        y_m = y1
        w_m = x2 - x1
        h_m = y2 - y1

        x_n = xx1
        y_n = yy1
        w_n = xx2 - xx1
        h_n = yy2 - yy1

        return [
            math.log(max((x_m - x_n), 1e-3) / w_m),
            math.log(max((y_m - y_n), 1e-3) / h_m),
            math.log(w_n / w_m),
            math.log(h_n / h_m),
        ]
    
    def get_feed_dict(self, batch,is_train=False):
        config = self.config
        # Tensor dimensions, so pylint: disable=g-bad-name
        N = self.N
        P = self.P
        KP = self.KP

        T_in = config.obs_len
        T_pred = config.pred_len

        feed_dict = {}

        # initial all the placeholder

        traj_obs_gt = torch.zeros(N, T_in, P, dtype=torch.float)
        traj_obs_gt_mask = torch.zeros(N, T_in, dtype=torch.bool)

        # for getting pred length during test time
        traj_pred_gt_mask = torch.zeros([N, T_pred], dtype=torch.bool)
        feed_dict["is_train"] = is_train

        data = batch.data
        assert len(data['obs_traj_rel']) == N

        for i, (obs_data, pred_data) in enumerate(zip(data['obs_traj_rel'],
                                                      data['pred_traj_rel'])):
            for j, xy in enumerate(
                    obs_data
            ):
                traj_obs_gt[i, j, :] = torch.from_numpy(xy)
                traj_obs_gt_mask[i, j] = True
            for j in range(config.pred_len):
                # used in testing to get the prediction length
                traj_pred_gt_mask[i, j] = True
                # ---------------------------------------

        # link the feed_dict
        feed_dict["traj_obs_gt"] = traj_obs_gt.to(self.device)
        feed_dict["traj_obs_gt_mask"] = traj_obs_gt_mask.to(self.device)
        feed_dict["traj_pred_gt_mask"] = traj_pred_gt_mask.to(self.device)

        # scene input
        obs_scene = torch.zeros((N, T_in), dtype=torch.int32)
        obs_scene_mask = torch.zeros((N, T_in), dtype=torch.bool)
        traj_pred_gt = torch.zeros(N, T_pred, P, dtype=torch.float)

        # each bacth
        for i in range(len(data['batch_obs_scene'])):
            for j in range(len(data['batch_obs_scene'][i])):
                # it was (1) shaped
                obs_scene[i, j] = data['batch_obs_scene'][i][j][0]
                obs_scene_mask[i, j] = True

        feed_dict["obs_scene"] = obs_scene.to(self.device)
        feed_dict["obs_scene_mask"] = obs_scene_mask.to(self.device)

        # [N,num_scale, T] # each is int to num_grid_class
        feed_dict['grid_pred_labels'] = [None] * len(config.scene_grids)
        feed_dict['grid_pred_targets'] = [None] * len(config.scene_grids)
        feed_dict['grid_obs_labels'] = [None] * len(config.scene_grids)
        # feed_dict['grid_obs_targets'] = [None] * len(config.scene_grids)
            
        for j, _ in enumerate(config.scene_grids):
            # [N, seq_len]
            # currently only the destination

            feed_dict['grid_obs_labels'].append(
                torch.empty(N, T_in, dtype=torch.int32))  # grid class
            this_grid_label = torch.zeros(N, T_in, dtype=torch.int32)
            
            for i in range(len(data['obs_grid_class'])):
                this_grid_label[i, :] = torch.from_numpy(data['obs_grid_class'][i][j, :])

            feed_dict["grid_obs_labels"][j] = this_grid_label

        feed_dict['grid_obs_labels'] = [i.to(self.device) for i in feed_dict['grid_obs_labels']]
        # feed_dict['grid_obs_targets'] = [i.to(self.device) for i in feed_dict['grid_obs_targets']]


        # person pose input
        if config.add_kp:
            obs_kp = torch.zeros(N, T_in, KP, 2, dtype=torch.float)

            # each bacth
            for i, obs_kp_rel in enumerate(data['obs_kp_rel']):
                for j, obs_kp_step in enumerate(obs_kp_rel):
                    obs_kp[i, j, :, :] = torch.from_numpy(obs_kp_step)

            feed_dict["obs_kp"] = obs_kp.to(self.device)

        if not self.config.preload_features:
            split = 'train'
            if not is_train:
                split = 'val'
            if config.is_test:
                split = 'test'

            # this is the h/w the bounding box is based on
            person_h = config.person_h
            person_w = config.person_w
            person_feat_dim = config.person_feat_dim

            obs_person_features = torch.zeros(
                N, T_in, person_h, person_w, person_feat_dim, dtype=torch.float)
            for i in range(len(data['obs_boxid'])):
                for j in range(len(data['obs_boxid'][i])):
                    boxid = data['obs_boxid'][i][j]
                    featfile = os.path.join(
                        config.person_feat_path, split, '%s.npy' % boxid)
                    obs_person_features[i, j] = torch.from_numpy(np.squeeze(
                        np.load(featfile), axis=0))
        else:
            obs_person_features = torch.from_numpy(data['obs_person_features'])
        feed_dict["obs_person_features"] = obs_person_features.to(self.device)

        # add other boxes,
        K = self.K  # max_other boxes
        other_boxes_class = torch.zeros(
            N, T_in, K, config.num_box_class, dtype=torch.float)
        other_boxes = torch.zeros(N, T_in, K, 4, dtype=torch.float)
        other_boxes_mask = torch.zeros(N, T_in, K, dtype=torch.bool)
        for i in range(len(data['obs_other_box'])):
            for j in range(len(data['obs_other_box'][i])):  # -> seq_len
                this_other_boxes = data['obs_other_box'][i][j]
                this_other_boxes_class = data['obs_other_box_class'][i][j]

                other_box_idxs = range(len(this_other_boxes))

                if config.random_other:
                    random.shuffle(other_box_idxs)

                other_box_idxs = other_box_idxs[:K]

                # get the current person box
                this_person_x1y1x2y2 = data['obs_box'][i][j]  # (4)

                for k, idx in enumerate(other_box_idxs):
                    other_boxes_mask[i, j, k] = True

                    other_box_x1y1x2y2 = this_other_boxes[idx]

                    other_boxes[i, j, k, :] = torch.tensor(self.encode_other_boxes(
                        this_person_x1y1x2y2, other_box_x1y1x2y2)).to(other_boxes)
                    # one-hot representation
                    box_class = this_other_boxes_class[idx]
                    other_boxes_class[i, j, k, box_class] = 1

        feed_dict['obs_other_boxes'] = other_boxes.to(self.device)
        feed_dict["obs_other_boxes_class"] = other_boxes_class.to(self.device)
        feed_dict["obs_other_boxes_mask"] = other_boxes_mask.to(self.device)

        if is_train:
            for i, (obs_data, pred_data) in enumerate(zip(data['obs_traj_rel'],
                                                          data['pred_traj_rel'])):
                for j, xy in enumerate(pred_data):
                    traj_pred_gt[i, j, :] = torch.from_numpy(xy)
                    traj_pred_gt_mask[i, j] = True

            for j, _ in enumerate(config.scene_grids):

                this_grid_label = torch.zeros(N, dtype=torch.int32)
                this_grid_target = torch.zeros(N, 2, dtype=torch.float)
                for i in range(len(data['pred_grid_class'])):
                    # last pred timestep
                    this_grid_label[i] = data['pred_grid_class'][i][j, -1]
                    # last pred timestep
                    this_grid_target[i] = torch.from_numpy(data['pred_grid_target'][i][j, -1])

                # add new label as kxk for more target loss?

                feed_dict['grid_pred_labels'][j] = this_grid_label
                feed_dict['grid_pred_targets'][j] = this_grid_target
            feed_dict['grid_pred_labels'] = [i.to(self.device) for i in feed_dict['grid_pred_labels']]
            feed_dict['grid_pred_targets'] = [i.to(self.device) for i in feed_dict['grid_pred_targets']]



        feed_dict["traj_pred_gt"] = traj_pred_gt.to(self.device)
        feed_dict["scene_feat"] = torch.tensor(data['batch_scene_feat']).to(self.device)

        if config.add_activity:
            future_act = torch.zeros(N, config.num_act, dtype=torch.uint8)
            # for experiment, training activity detection model

            for i in range(len(data['future_activity_onehot'])):
                future_act[i, :] = torch.from_numpy(data['future_activity_onehot'][i])

            feed_dict["future_act_label"] = future_act.to(self.device)

        # needed since it is in tf.conf, but all zero in testing
        feed_dict['traj_class_gt'] = torch.zeros(N, dtype=torch.int32)
        if config.multi_decoder and is_train:
            traj_class = torch.zeros(N, dtype=torch.int32)
            for i in range(len(data['traj_cat'])):
                traj_class[i] = data['traj_cat'][i]
            feed_dict["traj_class_gt"] = traj_class.to(self.device)
        return feed_dict


def reconstruct(tensor, ref, keep):
    """Reverse the flatten function.

      Args:
        tensor: the tensor to operate on
        ref: reference tensor to get original shape
        keep: index of dim to keep

      Returns:
        Reconstructed tensor
      """
    shape = list(ref.shape)
    shape[keep] = -1
    return tensor.reshape(*shape)


def softmax(logits, scope=None):
    """a flatten and reconstruct version of softmax."""
    flat_logits = logits.view(logits.shape[0], -1)
    flat_out = torch.softmax(flat_logits, dim=-1)
    out = reconstruct(flat_out, logits, 1)
    return out


def softsel(target, logits, use_sigmoid=False, scope=None):
    """Apply attention weights."""
    if use_sigmoid:
        a = torch.sigmoid(logits)
    else:
        a = softmax(logits)  # shape is the same
    target_rank = len(target.shape)
    # [N,M,JX,JQ,2d] elem* [N,M,JX,JQ,1]
    # second last dim
    return (torch.unsqueeze(a, -1) * target).sum(target_rank - 2)


def exp_mask(val, mask):
    """Apply exponetial mask operation."""
    return torch.add(val, (1 - mask.float()) * -1e30)


def focal_attention(query, context, use_sigmoid=False):
    """Focal attention layer.

  Args:
    query : [N, dim1]
    context: [N, num_channel, T, dim2]
    use_sigmoid: use sigmoid instead of softmax
    scope: variable scope

  Returns:
    Tensor
  """

    # Tensor dimensions, so pylint: disable=g-bad-name
    _, d = query.shape
    _, K, _, d2 = context.shape
    assert d == d2

    T = context.shape[2]

    # [N,d] -> [N,K,T,d]
    query_aug = torch.unsqueeze(
        torch.unsqueeze(query, 1), 1).repeat(1, K, T, 1)

    # cosine simi
    query_aug_norm = F.normalize(query_aug, p=2, dim=-1)
    context_norm = F.normalize(context, dim=-1)
    # [N, K, T]
    a_logits = torch.multiply(query_aug_norm, context_norm).sum(3)

    a_logits_maxed = a_logits.amax(2)  # [N,K]

    attended_context = softsel(softsel(context, a_logits,
                                       use_sigmoid=use_sigmoid), a_logits_maxed,
                               use_sigmoid=use_sigmoid)

    return attended_context


def concat_states(state_tuples, dim):
    """Concat LSTM states."""
    return LSTMStateTuple(c=torch.cat([s.c for s in state_tuples],
                                      dim=dim),
                          h=torch.cat([s.h for s in state_tuples],
                                      dim=dim))


class Trainer(object):
    """Trainer class for model."""

    def __init__(self, model, config):
        self.config = config
        self.model = model  # this is an model instance

        self.global_step = 1

        learning_rate = config.init_lr
        params = [
            {
                'params': v,
                'lr': learning_rate * config.emb_lr if "emb" in k else learning_rate,
            } for k, v in model.named_parameters()
        ]

        if config.optimizer == 'adadelta':
            self.opt = torch.optim.Adadelta(
                params, learning_rate
            )

        elif config.optimizer == 'adam':
            self.opt = torch.optim.Adam(
                params, learning_rate
            )
        else:
            raise Exception('Optimizer not implemented')

        if config.learning_rate_decay is not None:
            decay_steps = int(config.train_num_examples /
                              config.batch_size * config.num_epoch_per_decay)

            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.opt, gamma=config.learning_rate_decay,
                step_size=decay_steps
            )


    def step(self, batch):
        """One training step."""
        config = self.config
        self.model.train()
        # idxs is a tuple (23,123,33..) index for sample
        _, batch_data = batch
        batch_data = self.model.get_feed_dict(batch_data, is_train=True)
        out = self.model(batch_data)
        loss_dict, loss = self.model.cal_loss(batch_data, out)

        self.opt.zero_grad()
        loss.backward()

        if self.config.clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.clip_gradient_norm)
        self.opt.step()
        self.scheduler.step()
        act_loss = loss_dict.get('activity_loss', -1)
        xyloss = loss_dict.get('xyloss', -1)
        traj_class_loss = loss_dict.get('traj_class_loss', -1)
        grid_loss = []
        for i, _ in enumerate(config.scene_grids):
            grid_loss += [loss_dict.get(f'grid_class_loss_{i}', -1), \
                         loss_dict.get(f'grid_regression_loss_{i}', -1)]
        self.global_step += 1
        return loss, xyloss, act_loss, traj_class_loss, grid_loss


class Tester(object):
    """Tester for model."""

    def __init__(self, model, config):
        self.config = config
        self.model = model

    @torch.no_grad()
    def step(self, batch):
        """One inferencing step."""
        self.model.eval()
        _, batch_data = batch
        batch_data = self.model.get_feed_dict(batch_data)
        out = self.model(batch_data)
        out_np = {k: v.cpu().numpy() for k, v in out.items() if isinstance(v, torch.Tensor)}
        out_np['grid_target_logits'] = [i.detach().cpu().numpy() for i in out['grid_target_logits'] ]
        out_np['grid_class_logits'] = [i.detach().cpu().numpy() for i in out['grid_class_logits']]
        out = AttrDict(out_np)
        pred_out = out.get('traj_pred_out', None)
        future_act = out.get('future_act_logits', None)
        grid_pred_1, grid_pred_2 = out.get('grid_class_logits', None)
        # grid_pred_1, grid_pred_2 = out.get('grid_target_logits', None)
        traj_class_logits = out.get('traj_class_logits', None)


        traj_outs = out.get('traj_pred_out', None)

        return pred_out, future_act, grid_pred_1, grid_pred_2, traj_class_logits, \
               traj_outs
