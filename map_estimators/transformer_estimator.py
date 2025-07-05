import copy
import math
import tempfile
import numpy as np
import logging
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from tqdm import tqdm

from ..map_generators.map_generator import MapGenerator

from ..gsim.include.neural_net import NeuralNet, LossLandscapeConfig

from .neural_network_map_estimator import MapEstimator

import time

gsim_logger = logging.getLogger("gsim")


class TransformerDnnConf():

    def __init__(
            self,
            dim_input,
            num_heads,
            dim_embedding,
            num_layers=1,
            num_layers_encoder=None,  # by default ceil(num_layers/2)
            mlp_expansion_factor=4,
            dropout_prob=0.1,
            b_causal_masking=True,
            device_type=None):
        self.dim_input = dim_input
        self.num_heads = num_heads
        self.dim_embedding = dim_embedding

        self.mlp_expansion_factor = mlp_expansion_factor
        self.num_layers = num_layers
        self.num_layers_encoder = num_layers_encoder
        # dropout hyperparameters
        self.embd_pdrop = dropout_prob
        self.resid_pdrop = dropout_prob  # used in attn and mlp
        self.attn_pdrop = dropout_prob
        self.b_causal_masking = b_causal_masking
        self.device_type = device_type

    def override(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def clone(self):
        return copy.deepcopy(self)

    def copy(self, **kwargs):

        return self.clone().override(**kwargs)


class AttnMultiHead(nn.Module):

    def __init__(self, conf: TransformerDnnConf):
        super().__init__()

        self.num_heads = conf.num_heads

        self.k = nn.Linear(conf.dim_embedding, conf.dim_embedding)
        self.q = nn.Linear(conf.dim_embedding, conf.dim_embedding)
        self.v = nn.Linear(conf.dim_embedding, conf.dim_embedding)
        self.conf = conf

        self.attn_dropout = nn.Dropout(conf.attn_pdrop)
        self.b_causal_masking = conf.b_causal_masking
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        self.num_candidates = 0  # Set this property before training/evaluating. The reason why this is not passed to call is to avoid overriding several methods of NeuralNet.

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Args:

            `x`: batch_len x num_keys x dim_embedding. This provides the keys
            and the values.

            `y`: batch_len x num_queries x dim_embedding. If None, `y` is set to
            `x` (self attention).

            Masking or self.num_candidates>0 are only allowed with
            self-attention (y=None).

        Returns

            `t_val`: batch_len x num_queries x dim_embedding

            The last self.num_candidates embeddings do not receive attention
            from each other, only from the first num_queries -
            self.num_candidates embeddings. This is useful to compare one
            candidate with other, e.g. to obtain multiple estimates of the same
            target, each one using all the non-candidates and a single
            candidate.
        """

        def get_mask():
            """Returns a mask of shape num_embeddings x num_embeddings."""

            if self.num_candidates == 0 and not self.b_causal_masking:
                return None  # no mask in cross-attention

            # Get the mask for the non-candidate embeddings
            num_non_candidates = num_queries - self.num_candidates
            mask = torch.ones(num_non_candidates,
                              num_non_candidates,
                              dtype=torch.bool,
                              device=x.device)
            if self.b_causal_masking:
                mask = mask.tril_(diagonal=0)

            # Complete the mask if there are candidates
            if self.num_candidates:
                # The matrix has four blocks:
                #    - The top-left block is the above mask for the non-candidates
                #    - The top-right block is a matrix of zeros
                #    - The bottom-left block is a matrix of ones
                #    - The bottom-right block is the identity matrix of size self.num_candidates
                mask = torch.cat([
                    torch.cat([
                        mask,
                        torch.zeros(num_non_candidates,
                                    self.num_candidates,
                                    dtype=torch.bool,
                                    device=x.device)
                    ],
                              dim=1),
                    torch.cat([
                        torch.ones(self.num_candidates,
                                   num_non_candidates,
                                   dtype=torch.bool,
                                   device=x.device),
                        torch.eye(self.num_candidates,
                                  dtype=torch.bool,
                                  device=x.device)
                    ],
                              dim=1)
                ],
                                 dim=0)

            return mask

        def do_masking(t_qk):
            """            
            Args:
                `t_qk` is of shape batch_len x num_embeddings x num_embeddings.

            Returns:
                `t_qk_masked`: tensor resulting from applying the mask
            """
            if self.b_causal_masking or self.num_candidates:
                # num_embeddings = t_qk.shape[-1]
                # m_mask = torch.tril(
                #     torch.ones(
                #         (num_embeddings, num_embeddings))).to(t_qk.device)
                m_mask = get_mask()
                return torch.masked_fill(t_qk, m_mask == 0, -torch.inf)
            else:
                return t_qk

        def reshape_for_multihead_attention(t_in):
            num_embeddings = t_in.shape[-2]
            return t_in.view(
                batch_len, num_embeddings, self.num_heads,
                dim_embedding // self.num_heads
            ).transpose(
                1, 2
            )  # batch_len x num_heads x num_embeddings x dim_embedding/num_heads

        if y is not None:
            # Cross attention
            assert self.num_candidates == 0
            assert not self.b_causal_masking
        else:
            # Self attention
            y = x

        batch_len, num_queries, dim_embedding = y.shape

        t_k = self.k(x)  # batch_len x num_keys x dim_embedding
        t_v = self.v(x)
        t_q = self.q(y)  # batch_len x num_queries x dim_embedding

        # batch_len x num_heads x num_embeddings x dim_embedding/num_heads
        t_k = reshape_for_multihead_attention(
            t_k)  # batch_len x num_heads x num_keys x dim_embedding/num_heads
        t_v = reshape_for_multihead_attention(t_v)  # same
        t_q = reshape_for_multihead_attention(
            t_q
        )  # batch_len x num_heads x num_queries x dim_embedding/num_heads

        if self.flash:
            t_v_heads = torch.nn.functional.scaled_dot_product_attention(
                t_q,
                t_k,
                t_v,
                attn_mask=get_mask(),
                dropout_p=self.conf.attn_pdrop,
                is_causal=False
            )  # batch_len x num_heads x num_queries x dim_embedding/num_heads

        else:
            t_qk = t_q @ t_k.transpose(
                -2, -1)  # batch_len x num_heads x num_queries x num_keys
            scale = 1 / ((dim_embedding // self.num_heads)**0.5)
            t_weights = F.softmax(
                do_masking(t_qk) * scale,
                -1)  # batch_len x num_heads x num_queries x num_keys

            t_weights = self.attn_dropout(t_weights)

            t_v_heads = t_weights @ t_v  # batch_len x num_heads x num_queries x dim_embedding/num_heads

        return t_v_heads.transpose(1, 2).contiguous().view(
            batch_len, num_queries, dim_embedding)


class AttnBlock(nn.Module):

    class NewGELU(nn.Module):
        """
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
        Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
        """

        def forward(self, x):
            return 0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def __init__(self, conf: TransformerDnnConf, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ln_1 = nn.LayerNorm(conf.dim_embedding)
        self.att = AttnMultiHead(conf)
        self.ln_2 = nn.LayerNorm(conf.dim_embedding)
        self.mlp = nn.ModuleDict(
            dict(
                fc_1=nn.Linear(conf.dim_embedding,
                               conf.mlp_expansion_factor * conf.dim_embedding),
                gelu=AttnBlock.NewGELU(),
                fc_2=nn.Linear(conf.mlp_expansion_factor * conf.dim_embedding,
                               conf.dim_embedding),
                dropout=nn.Dropout(conf.resid_pdrop),
            ))

    def mlp_forward(self, x):

        return self.mlp.dropout(self.mlp.fc_2(self.mlp.gelu(self.mlp.fc_1(x))))

    def forward(self, x):
        x = x + self.att(self.ln_1(x))
        x = x + self.mlp_forward(self.ln_2(x))
        return x


class TransformerDnn(NeuralNet):

    def __init__(self, conf: TransformerDnnConf, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if conf.device_type is not None:
            self.device_type = conf.device_type
        self.obs_embedding = nn.Linear(
            conf.dim_input, conf.dim_embedding).to(device=self.device_type)
        self.loc_embedding = nn.Linear(
            conf.dim_input - 1, conf.dim_embedding).to(device=self.device_type)

        self.drop = nn.Dropout(conf.embd_pdrop).to(device=self.device_type)

        num_layers_encoder = int(np.ceil(
            conf.num_layers /
            2)) if conf.num_layers_encoder is None else conf.num_layers_encoder

        self.obs_encoder = nn.Sequential(*[
            AttnBlock(conf).to(device=self.device_type)
            for _ in range(num_layers_encoder)
        ])
        self.obs_decoder = nn.Sequential(*[
            AttnBlock(conf).to(device=self.device_type)
            for _ in range(conf.num_layers - num_layers_encoder)
        ])

        conf_no_masking = conf.copy(b_causal_masking=False)

        self.loc_encoder = nn.Sequential(*[
            AttnBlock(conf_no_masking).to(device=self.device_type)
            for _ in range(num_layers_encoder)
        ])
        self.cross_att_combiner = AttnMultiHead(conf_no_masking).to(
            device=self.device_type)
        self.loc_decoder = nn.Sequential(*[
            AttnBlock(conf_no_masking).to(device=self.device_type)
            for _ in range(max(0, conf.num_layers - num_layers_encoder - 1))
        ])

        # self.l_blocks = nn.ModuleList([
        #     AttnBlock(conf).to(device=self.device_type)
        #     for _ in range(conf.num_layers)
        # ])
        self.b_causal_masking = conf.b_causal_masking

        gsim_logger.info(
            f'Num. params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}'
        )

        self.initialize()

        self._num_candidates = 0
        self.num_candidates_train = None

    @property
    def num_candidates(self):
        return self._num_candidates

    @num_candidates.setter
    def num_candidates(self, num_candidates):
        self._num_candidates = num_candidates
        for block in list(self.obs_encoder) + list(self.obs_decoder):
            block.att.num_candidates = num_candidates

    def forward(self, t_data):
        """

        If self.num_candidates == 0:
            
            Args:
                `t_data`: batch_size x num_obs x dim_input

            Returns: 
                `m_pred`: batch_size x num_obs
        
        If self.num_candidates > 0:

            Args:
                `t_data`: batch_size x num_obs + self.num_candidates x dim_input

            Returns:
                `m_pred`: batch_size x num_obs + self.num_candidates +
                self.num_candidates. 
                
                    m_pred[:,:num_obs] are the estimates given the
                    non-candidates. 
                    
                    m_pred[:,num_obs:-self.num_candidates] are the  estimates
                    given all the non-candidates and the corresponding
                    candidate. 
                    
                    m_pred[:,-self.num_candidates:] are the weights of the
                    candidates.


        """
        t_obs_embeddings = self.obs_embedding(t_data.to(
            self.device_type))  # batch_size x num_obs x dim_embedding

        t_obs_embeddings = self.drop(t_obs_embeddings)

        t_obs_code = self.obs_encoder(t_obs_embeddings)
        t_obs_out = self.obs_decoder(
            t_obs_code)  #  batch_size x num_obs x dim_embedding
        m_est = torch.mean(t_obs_out, dim=-1)  # batch_size x num_points

        if self.num_candidates:
            t_candidates = t_data[:, -self.num_candidates:]
            t_candidates_loc = t_candidates[:, :,
                                            1:]  # only the features, not the measurement values
            t_loc_embeddings = self.loc_embedding(
                t_candidates_loc.to(self.device_type))
            t_loc_code = self.loc_encoder(t_loc_embeddings)
            t_obs_loc_code = self.cross_att_combiner(
                t_obs_code[:, :-self.num_candidates],
                t_loc_code)  # batch_size x num_candidates x dim_embedding
            m_obs_loc_out = self.loc_decoder(t_obs_loc_code).mean(
                dim=-1)  # batch_size x num_candidates
            m_weights = torch.softmax(m_obs_loc_out,
                                      dim=1)  # batch_size x num_candidates

            return torch.cat([m_est, m_weights], dim=1)

        return m_est

        # return torch.mean(t_embeddings[:, -1, :], dim=[1],
        #                   keepdim=True)  # batch_size x 1

    def get_combined_est(self, m_out):
        """
        Args:
            `m_out`: batch_size x num_obs + self.num_candidates +
            self.num_candidates is the tensor produced by forward.

        Returns:
            The combined estimate of the candidates. (batch_size,)
        
        """
        assert self.num_candidates > 0

        m_weights = m_out[:,
                          -self.num_candidates:]  # batch_size x num_candidates
        m_candidate_est = m_out[:, -2 * self.num_candidates:-self.
                                num_candidates]  # batch_size x num_candidates
        v_combined_est = torch.sum(m_weights * m_candidate_est,
                                   dim=1)  # batch_size,

        return v_combined_est

    def get_f_loss(self, min_context_len):

        if self.b_causal_masking:
            # In this case, all the outputs are meaningful
            assert min_context_len > 0
            first_ind_to_include = min_context_len - 1
        else:
            # In this case, only the last non-candidate embedding is meaningful
            assert min_context_len == 1
            first_ind_to_include = -1

        f_loss = lambda m_preds, m_targets: torch.mean(
            torch.square(m_preds[:, first_ind_to_include:] -
                         m_targets[:, first_ind_to_include:]),
            dim=1)  # The candidate embeddings are always included

        return f_loss

    @property
    def name_on_figs(self):
        return f"{self.__class__.__name__}"

    def split_output(self, m_out):
        """
        Args:
        
            `m_output` is the matrix returned by `forward`.
        
        Returns:
            A dict with keys:

            `m_obs_est`: estimates using the non-candidates.

            `m_weights`: weights of the candidates.

            `m_candidate_est`: estimates using the candidates.

        """

        if self.num_candidates == 0:
            return m_out, None, None

        num_obs = m_out.shape[1] - 2 * self.num_candidates

        return {
            "m_obs_est": m_out[:, :num_obs],
            "m_weights": m_out[:, -self.num_candidates:],
            "m_candidate_est": m_out[:, num_obs:-self.num_candidates]
        }

    def _get_loss(self, m_data, f_loss):
        m_feat_batch, v_targets_batch = m_data
        m_feat_batch = m_feat_batch.float().to(self.device_type)
        v_targets_batch = v_targets_batch.float().to(self.device_type)

        if self.num_candidates_train is None:
            v_targets_batch_pred = self(m_feat_batch.float())
            loss = f_loss(v_targets_batch_pred.float(),
                          v_targets_batch.float())
        else:

            def f_loss_combined_est(m_preds, m_targets):
                return torch.square(
                    self.get_combined_est(m_preds) - m_targets[:, 0])

            loss = torch.zeros(m_feat_batch.shape[0], device=self.device_type)
            for num_candidates in self.num_candidates_train:
                self.num_candidates = num_candidates
                m_preds = self(m_feat_batch.float())
                m_obs_est = self.split_output(m_preds)["m_obs_est"]

                loss += f_loss(
                    m_obs_est.float(), v_targets_batch[:, :m_obs_est.shape[1]]
                ) + f_loss_combined_est(m_preds, v_targets_batch.float())
            loss = loss / len(self.num_candidates_train) / 2

        assert loss.shape[0] == m_feat_batch.shape[
            0] and loss.ndim == 1, "f_loss must return a vector of length batch_size."
        return loss


class AttnMapEstimator(MapEstimator):

    name_on_figs = "AttnMapEstimator"
    estimation_fun_type = 's2s'

    def __init__(self,
                 num_feat,
                 att_dnn: TransformerDnn,
                 batch_size_to_estimate=1024):
        """
        `att_dnn` is a DNN that takes tensors of shape batch_len x num_pts x
        num_feat + 1 and returns matrices of length batch_len x num_pts.

        The last self.num_candidates are used to evaluate possible new
        measurement locations. 
        
        """
        self.att_dnn = att_dnn
        self.num_feat = num_feat
        self.batch_size_to_estimate = batch_size_to_estimate
        self._num_candidates = 0

    @property
    def num_candidates(self):
        return self._num_candidates

    @num_candidates.setter
    def num_candidates(self, num_candidates):
        self._num_candidates = num_candidates
        self.att_dnn.num_candidates = num_candidates

    @staticmethod
    def _adapt_input(m_loc, v_meas, m_target_loc, num_feat):
        """
        Args:
            `m_loc`: num_obs x 3
            `v_meas`: num_obs
            `m_target_loc`: num_target_locs x 3

        Performs in a batch (of size num_target_locs) the following. For each
        target location, 
            . Translate the observations so that the target location is at the origin.
            . Rotate the observations so that the x axis points towards
                    sum_n exp(v_meas[n]) * (m_loc[n] - target_loc),
                where n is the index of the n-th observation.
                This rotation angle is referred to as `theta*`.

        After rotating, the features will be:
            . The measurement values.
            . The translated coordinates of the observations.
            . theta*.
            . cos(theta[n]), where theta[n] is the angle between the x axis and the n-th observation.
            . sin(theta[n]).
            . The distance to the target location.
            . 1/(distance + 1e-1).
            . (1/(distance + 1e-1))^2.

        Returns:
            `t_feat`: num_target_locs x num_obs x num_feat+1 
        """

        def get_angle(m_direction):
            """
            Args:
                `m_direction`: num_target_locs x 3
            """

            return np.arctan2(m_direction[:, 1], m_direction[:, 0])

        def rotate_cter_clkwise(t_pre_dist, v_angles):
            """
                Args:
                    `t_pre_dist`: num_target_locs x num_points x 3
                    `angle`: rad
            """
            t_cos = np.cos(-v_angles)[:, None, None]
            t_sin = np.sin(-v_angles)[:, None, None]
            t_zeros = np.zeros_like(v_angles)[:, None, None]
            t_ones = np.ones_like(v_angles)[:, None, None]
            t_rotation = np.concatenate([
                t_cos, -t_sin, t_zeros, t_sin, t_cos, t_zeros, t_zeros,
                t_zeros, t_ones
            ],
                                        axis=1).reshape(-1, 3, 3)

            return t_pre_dist @ t_rotation.transpose(0, 2, 1)

        def get_angles(t_pre_dist):
            return np.arctan2(t_pre_dist[:, :, 1], t_pre_dist[:, :, 0])

        num_target_locs = m_target_loc.shape[0]

        t_pre_dist = np.tile(
            m_loc[None, ...],
            (num_target_locs, 1,
             1)) - m_target_loc[:, None, :]  # num_target_locs x num_meas x 3

        v_not_nans = ~np.isnan(v_meas)
        min_meas = np.min(v_meas[v_not_nans])

        v_weights = np.exp(v_meas - min_meas)  # num_meas

        m_direction = np.sum(t_pre_dist[:, v_not_nans, :] *
                             np.tile(v_weights[None, v_not_nans, None],
                                     (num_target_locs, 1, 1)),
                             axis=1) / np.sum(v_weights[v_not_nans])
        v_angles = get_angle(m_direction)

        t_pre_dist = rotate_cter_clkwise(t_pre_dist, v_angles)

        t_meas = np.tile(v_meas[None, :, None], (num_target_locs, 1, 1))

        t_feat_0 = t_pre_dist[:, :, [0]]
        t_feat_1 = t_pre_dist[:, :, [1]]
        t_feat_2 = get_angles(t_pre_dist)[..., None]
        t_feat_3 = np.cos(t_feat_2)
        t_feat_4 = np.sin(t_feat_2)

        t_feat_5 = np.sqrt(np.sum(np.square(t_pre_dist), axis=2))[..., None]
        t_feat_6 = 1 / (t_feat_5 + 1e-1)
        t_feat_7 = np.square(t_feat_6)

        return np.concatenate(
            [
                t_meas, t_feat_0, t_feat_1, t_feat_2, t_feat_3, t_feat_4,
                t_feat_5, t_feat_6, t_feat_7
            ],
            axis=2)[:, :, :num_feat +
                    1]  # num_target_locs x num_meas x num_feat+1

    @staticmethod
    def gen_tensor_dataset(t_locs, m_meas, num_feat):

        l_inputs = []
        l_outputs = []

        print("Preprocessing data...")
        for ind_example, v_rx_power in enumerate(m_meas):

            v_x_target = t_locs[ind_example, 0]
            m_locs_input = t_locs[ind_example, 1:]
            v_y_input = v_rx_power[1:]

            l_inputs.append(
                AttnMapEstimator._adapt_input(m_locs_input, v_y_input,
                                              v_x_target[None, :],
                                              num_feat)[0])
            l_outputs.append(np.tile(v_rx_power[[0]], m_meas.shape[1] - 1))

        dataset = TensorDataset(torch.from_numpy(np.array(l_inputs)),
                                torch.from_numpy(np.array(l_outputs)))

        return dataset

    def train(self,
              dataset_train,
              dataset_val,
              lr,
              min_context_len=1,
              num_candidates_train=None,
              **kwargs):
        """
        Args:
            `dataset_train`: TensorDataset. In training example i,
                `dataset_train[i][0]`: num_meas x num_feat+1
                `dataset_train[i][1]`: num_meas

            `num_candidates`: See AttnMultiHead.forward
        """

        assert min_context_len < dataset_train[0][1].shape[0]

        self.att_dnn.num_candidates_train = num_candidates_train

        return self.att_dnn.fit(
            dataset=dataset_train,
            dataset_val=dataset_val,
            optimizer=torch.optim.AdamW(self.att_dnn.parameters(), lr=lr),
            f_loss=self.att_dnn.get_f_loss(min_context_len),
            **kwargs)

    def estimate_s2s(
            self,
            measurement_loc=None,
            measurements=None,
            building_meta_data=None,  # unused
            test_loc=None):
        """
        Args:
            - `measurement_locs` : num_measurements x 3 matrix with the
                   3D locations of the measurements.
            - `measurements` : num_measurements x num_sources matrix
                   with the measurements at each channel.
            - `test_loc`: a num_test_loc x 3 matrix with the
                    locations where the map estimate will be evaluated.

        Returns:
            `d_map_estimate`: dictionary whose fields are:

           - "t_power_map_estimate" :  num_test_loc x num_sources matrix with
                estimated power of each channel at test_loc.
           - Other optional keys.
        """

        if measurements.shape[1] > 1:
            raise NotImplementedError("Only one source is supported")

        v_y = measurements[:, 0]

        lv_estimates = []
        for ind in range(0, test_loc.shape[0], self.batch_size_to_estimate):
            t_data = self._adapt_input(
                measurement_loc, v_y,
                test_loc[ind:ind + self.batch_size_to_estimate], self.num_feat)

            m_out = self.att_dnn(
                torch.tensor(t_data).float()).cpu().detach().numpy()
            if self.num_candidates:
                v_est = self.att_dnn.get_combined_est(m_out)
            else:
                v_est = m_out[:, -1]
            lv_estimates.append(v_est)

        m_power_map_estimate = np.concatenate(lv_estimates)

        return {"t_power_map_estimate": m_power_map_estimate[:, None]}

    @staticmethod
    def generate_dataset(map_generator: MapGenerator,
                         num_meas_per_map: int,
                         num_examples_per_map: int,
                         num_patches_to_gen: int,
                         sampling_mode="uniform_standard"):
        """
        Let num_examples = num_patches_to_gen * num_examples_per_map
        
        Returns:
            `t_locs_train`: a (num_examples x num_meas_per_map x 3) tensor
                storing 3D locations.

            `m_meas_train`: a (num_examples x num_meas_per_map) matrix of the
                measurements at those locations.
        """
        lm_meas_locs = []
        lv_meas = []

        for ind_real in tqdm(range(num_patches_to_gen)):

            attempts = 0
            while True:
                patch_map = map_generator.generate_map()
                attempts += 1
                if patch_map.get_num_meas(sampling_mode) >= num_meas_per_map:
                    if attempts > 20:
                        print(f'Generating a map required {attempts} attempts')
                    break

            for _ in range(num_examples_per_map):

                patch_map_obs = patch_map.get_obs(sampling_mode=sampling_mode,
                                                  num_obs=num_meas_per_map,
                                                  return_nobs=False)
                lm_meas_locs.append(patch_map_obs.m_meas_locs_sf)
                lv_meas.append(patch_map_obs.m_meas_sf[:, 0])

            ind_real = ind_real + num_examples_per_map

        return np.array(lm_meas_locs), np.array(lv_meas)

    def estimate_and_select_next_loc(self,
                                     t_test_loc,
                                     t_obs_locs,
                                     t_obs_meas,
                                     t_candidate_locs,
                                     t_candidate_meas=None):
        """
        Args:
            `t_test_loc`: num_examples x 3 
            
            `t_obs_locs`: num_examples x num_meas x 3

            `t_obs_meas`: num_examples x num_meas x 1 
            
            `t_candidate_locs`: num_examples x num_candidates x 3 
            
            `t_candidate_meas`: num_examples x num_candidates x 1 [optional]

        Returns:

            A dict with the following keys and values:

            - "obs_est": num_examples x 1. The n-th entry is the estimate of the
              map at t_test_loc[n] given the observations in t_obs_locs[n] and
              t_obs_meas[n].

            - "candidate_weights": num_examples x num_candidates. The n-th entry
                is the weight of each candidate location in t_candidate_locs[n]
                when estimating the map at t_test_loc[n].

            - "next_loc": num_examples vector. The n-th entry is the index of 
                the candidate location in t_candidate_locs[n] that is selected
                as the next measurement location.

            If t_candidate_meas is provided, the following keys are included:

            - "candidate_estimates": num_examples x num_candidates. Only present
               The m,n-th entry contains the 
              estimates of the map at t_test_loc[m] given the measurements in
                t_obs_locs[m] and t_obs_meas[m] and the candidate measurement in
                t_candidate_locs[m,n] and t_candidate_meas[m,n].

            - "combined_candidate_estimates": num_examples x 1. The n-th entry
                contains the estimate of the map at t_test_loc[n] obtained by
                combining the estimates in "candidate_estimates" using the
                weights in "candidate_weights".

            - "candidate_estimate_with_greatest_weight": num_examples x 1. The
                n-th entry contains candidate_estimates[n, next_loc[n]].
                
        """

        num_examples, num_obs, _ = t_obs_locs.shape
        _, num_candidates, _ = t_candidate_locs.shape

        assert num_candidates > 0

        num_channels = t_obs_meas.shape[2]
        if num_channels > 1:
            raise NotImplementedError("Only one source is supported")
        ind_channel = 0

        no_candidate_meas = t_candidate_meas is None
        if no_candidate_meas:
            t_candidate_meas = torch.full((num_examples, num_candidates, 1),
                                          torch.nan)

        t_locs = np.concatenate(
            [t_obs_locs, t_candidate_locs],
            axis=1)  # num_examples x (num_meas + num_candidates) x 3
        t_meas = np.concatenate(
            [t_obs_meas, t_candidate_meas],
            axis=1)  # num_examples x (num_meas + num_candidates) x 1

        t_input = np.concatenate([
            self._adapt_input(t_locs[ind], t_meas[ind, :, ind_channel],
                              t_test_loc[[ind], :], self.num_feat)
            for ind in range(num_examples)
        ],
                                 axis=0)
        self.num_candidates = num_candidates

        if no_candidate_meas:
            t_input[:, -num_candidates:,
                    0] = 0  # The nans need to be replaced with an arbitrary number because 0*nan = nan, so masking does not work in the network if there are nans. Just don't use the estimates of the candidates.
        m_out = self.att_dnn(torch.tensor(t_input).float(
        ))  # num_examples x (num_meas + num_candidates) + num_candidates

        d_net_out = self.att_dnn.split_output(m_out.cpu().detach().numpy())

        d_out = {
            "obs_est": d_net_out["m_obs_est"][:, [-1]],
            "candidate_weights": d_net_out["m_weights"],
            "next_loc": np.argmax(d_net_out["m_weights"], axis=1)
        }

        if not no_candidate_meas:
            d_out["candidate_estimates"] = d_net_out["m_candidate_est"]
            d_out[
                "combined_candidate_estimates"] = self.att_dnn.get_combined_est(
                    m_out)
            d_out["candidate_estimate_with_greatest_weight"] = np.array([
                d_net_out["m_candidate_est"][ind, next_loc]
                for ind, next_loc in enumerate(d_out["next_loc"])
            ])

        return d_out
