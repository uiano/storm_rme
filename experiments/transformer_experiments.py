import inspect
import os
import numpy as np
import torch

import tensorflow as tf
import sklearn
from collections import OrderedDict
from tqdm import tqdm

from ..map_estimators.transformer_estimator import AttnMapEstimator, TransformerDnn, TransformerDnnConf

from ..map_generators.insite_map_generator import InsiteMapGenerator
from ..map_generators.map import Map
from ..map_generators.grid import RectangularGrid
from ..map_generators.real_data_map_generator import RealDataMapGenerator

from ..map_estimators.interpolation_estimators import (
    KernelRidgeRegressionEstimator, KNNEstimator)

from ..map_estimators.kriging_estimator import (GudmundsonBatchKrigingEstimator
                                                )
from ..map_estimators.neural_network_map_estimator import (
    NeuralNetworkEstimator, NeuralNetworkMapEstimator)
from ..map_estimators.std_aware_nn_estimators import (
    SurveyingStdAwareNnEstimator, UnetStdAwareNnEstimator)
from ..map_estimators.transformer_estimator import AttnMapEstimator

from ..map_estimators.std_unaware_nn_estimators import (
    RadioUnetEstimator, CompletionAutoencoderEstimator)

import pickle

import gsim
from gsim.gfigure import GFigure
from gsim_conf import folder_datasets

from ..gsim.include.neural_net import NeuralNet, LossLandscapeConfig

from ..simulators.map_estimation_simulator import MapEstimationSimulator

from ..map_estimators.transformer_estimator import AttnMapEstimator
from torch.utils.data import TensorDataset

dnn_folder = 'output/trained_estimators/transformers/'


def get_non_preprocessed_file_base_name(num_meas_per_map):
    return f'non-preprocessed_transformer_dataset-{num_meas_per_map}_meas_per_map.pkl'


def get_preprocessed_file_base_name(num_meas, num_feat):
    return f'preprocessed_transformer_dataset-{num_meas}_meas-{num_feat}_feats.pth'


def gen_dataset(map_generator,
                save_to_folder,
                num_obs,
                num_examples_per_map,
                num_patches_to_gen,
                num_feat=None,
                preprocess=True):
    """
    Args:
        - preprocess_dataset (bool): If True, the features are computed and the
          dataset is stored as a PyTorch TensorDataset. If False, the dataset
          is stored as a dictionary with keys 't_locs' and 'm_meas', where
          't_locs' is a tensor of shape (num_examples, num_obs + 1, 3) and
          'm_meas' is a tensor of shape (num_examples, num_obs + 1).
    
    """

    num_meas_per_map = num_obs + 1

    t_locs_train, m_meas_train = AttnMapEstimator.generate_dataset(
        map_generator=map_generator,
        num_meas_per_map=num_meas_per_map,
        num_examples_per_map=num_examples_per_map,
        num_patches_to_gen=num_patches_to_gen)

    print(f'Num. train. examples: {t_locs_train.shape[0]}')
    print(f'Num. obs: {t_locs_train.shape[1]}')

    # # save the dataset
    if not preprocess:
        file_path = save_to_folder + get_non_preprocessed_file_base_name(
            num_meas_per_map)
        if not os.path.exists(save_to_folder):
            os.makedirs(save_to_folder)
        with open(file_path, 'wb') as f:
            d_dataset = {'t_locs': t_locs_train, 'm_meas': m_meas_train}
            pickle.dump(d_dataset, f)

        # t_locs_train = t_locs_train[:, :num_obs]
        # m_meas_train = m_meas_train[:, :num_obs]

    else:

        dataset = AttnMapEstimator.gen_tensor_dataset(t_locs_train,
                                                      m_meas_train, num_feat)

        os.makedirs(save_to_folder, exist_ok=True)
        file_path = os.path.join(
            save_to_folder,
            get_preprocessed_file_base_name(num_meas_per_map, num_feat))

        save_tensor_dataset(dataset, file_path)

    return file_path


def load_dataset(load_from_folder):
    with open(load_from_folder + 'transformer_dataset.pkl', 'rb') as f:
        d_dataset = pickle.load(f)
        t_locs_train = d_dataset['t_locs']
        m_meas_train = d_dataset['m_meas']
    return t_locs_train, m_meas_train


def train_transformer(dataset_train,
                      dataset_val,
                      nn_folder=None,
                      num_candidates_train=None,
                      context_len_min=1,
                      num_feat=2,
                      num_heads=4,
                      dim_embedding=64,
                      num_layers=10,
                      dropout_prob=0.,
                      b_causal_masking=False,
                      device_type='mps',
                      num_epochs=800,
                      val_split=0.2,
                      lr=1e-2,
                      batch_size=64,
                      batch_size_eval=1024,
                      lr_patience=1024,
                      lr_decay=0.5,
                      shuffle=True,
                      first_epoch_to_plot=0,
                      llc=LossLandscapeConfig(
                          epoch_inds=[],
                          max_num_directions=4,
                          neg_gradient_step_scales=np.linspace(
                              -2e-3, 2e-3, 19))):

    np.random.seed(101)  #100
    torch.manual_seed(101)

    ade = AttnMapEstimator(
        num_feat=num_feat,
        att_dnn=TransformerDnn(
            TransformerDnnConf(
                dim_input=num_feat + 1,  # num_feat + 1                
                num_heads=num_heads,
                dim_embedding=dim_embedding,
                num_layers=num_layers,
                dropout_prob=dropout_prob,
                b_causal_masking=b_causal_masking,
                device_type=device_type,
            ),
            nn_folder=nn_folder,
        ))

    # train the models
    G_train_ade = NeuralNet.plot_training_history(
        ade.train(min_context_len=context_len_min,
                  num_candidates_train=num_candidates_train,
                  val_split=val_split,
                  dataset_train=dataset_train,
                  dataset_val=dataset_val,
                  lr=lr,
                  num_epochs=num_epochs,
                  batch_size=batch_size,
                  batch_size_eval=batch_size_eval,
                  lr_patience=lr_patience,
                  lr_decay=lr_decay,
                  shuffle=shuffle,
                  best_weights=True,
                  llc=llc),
        first_epoch_to_plot=first_epoch_to_plot)

    return G_train_ade


def save_tensor_dataset(dataset, file_path):
    data_dict = {'tensors': dataset.tensors}
    torch.save(data_dict, file_path)


def load_tensor_dataset(file_path):
    data_dict = torch.load(file_path, weights_only=True)
    return TensorDataset(*data_dict['tensors'])


def init_map_generator(patch_side_len=None,
                       num_examples_per_patch=None,
                       num_patches_train=None,
                       num_patches_test=None,
                       dataset=None):
    """
    Args:
        dataset (str): 'usrp', '4g', or 'ray-tracing'
    
    Returns:
        map_generator (MapGenerator)
        folder_train_dataset (str)
        folder_test_dataset (str)
        
    """
    num_examples_train = num_examples_per_patch * num_patches_train
    num_examples_test = num_examples_per_patch * num_patches_test

    if dataset == 'usrp':
        folder_train_dataset = f'output/datasets/usrp_train_{int(num_examples_train/1000)}k/'
        folder_test_dataset = f'output/datasets/usrp_test_{int(num_examples_test/1000)}k/'

        map_generator = RealDataMapGenerator(
            l_file_num=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            patch_side_len=patch_side_len,
            z_coord=5,
            folder=folder_datasets +
            'rme_datasets/usrp_data/grid_spacing_120_cm-freq_918_MHz/')
    elif dataset == '4g':
        folder_train_dataset = f'output/datasets/gradiant_train_{int(num_examples_train/1000)}k-patch_{patch_side_len}_m/'
        folder_test_dataset = f'output/datasets/gradiant_test_{int(num_examples_test/1000)}k-patch_{patch_side_len}_m/'

        map_generator = RealDataMapGenerator(
            l_file_num=[0, 1, 2],
            patch_side_len=patch_side_len,
            folder=folder_datasets + 'rme_datasets/gradiant/combined/rsrp')
    elif dataset == 'ray-tracing':
        folder_train_dataset = f'output/datasets/insite_train_{int(num_examples_train/1000)}k/'
        folder_test_dataset = f'output/datasets/insite_test_{int(num_examples_test/1000)}k/'

        map_generator = InsiteMapGenerator(
            l_file_num=list(range(1, 6)) +
            list(range(8, 41)),  # use 41 and 42 for the test set
            patch_side_len=patch_side_len,
            #     z_coord=5, # not considered so far. Useful?
            folder=folder_datasets + "insite_data/power_rosslyn/",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return map_generator, folder_train_dataset, folder_test_dataset


def test_transformers_on_active_sensing(num_meas=None,
                                        attn_estimator=None,
                                        v_num_obs_test=None,
                                        folder_test_dataset=None):
    """
        Returns a GFigure: RMSE vs. number of observations
    """

    # load the test dataset
    file_name = get_non_preprocessed_file_base_name(num_meas)
    with open(os.path.join(folder_test_dataset, file_name), 'rb') as f:
        d_dataset = pickle.load(f)
        t_locs = d_dataset['t_locs']  # num_examples x num_meas x 3
        m_meas = d_dataset['m_meas']  # num_examples x num_meas

    def get_error(t_locs, m_meas, num_obs, num_iter=None):

        num_candidates = t_locs.shape[1] - num_obs - 1
        assert num_candidates >= 0

        def split_meas(t):
            return t[:, 0, :], t[:, 1:1 + num_obs, :], t[:, 1 + num_obs:, :]

        l_err_obs = []
        l_err_selected_candidate = []
        l_err_non_selected_candidates = []

        num_examples_train = num_iter if num_iter is not None else t_locs.shape[
            0]
        assert num_examples_train <= t_locs.shape[0]
        for ind_example in tqdm(range(num_examples_train)):
            t_test_loc, t_obs_locs, t_candidate_locs = split_meas(
                t_locs[[ind_example], ...])
            t_test_meas, t_obs_meas, t_candidate_meas = split_meas(
                m_meas[[ind_example], :, None])

            d_out = attn_estimator.estimate_and_select_next_loc(
                t_test_loc=t_test_loc,
                t_obs_locs=t_obs_locs,
                t_obs_meas=t_obs_meas,
                t_candidate_locs=t_candidate_locs,
                t_candidate_meas=t_candidate_meas,
            )

            l_err_obs.append((t_test_meas - d_out['obs_est'])[0, 0]**2)
            l_err_selected_candidate.append(
                (t_test_meas -
                 d_out['candidate_estimate_with_greatest_weight'])[0, 0]**2)
            l_err_non_selected_candidates += list(
                (t_test_meas[0, 0] - np.array([
                    d_out['candidate_estimates'][0, ind]
                    for ind in range(num_candidates)
                    if ind != d_out["next_loc"][0]
                ]))**2)

        return {
            'err_obs':
            np.sqrt(np.mean(l_err_obs)),
            'err_selected_candidate':
            np.sqrt(np.mean(l_err_selected_candidate)),
            'err_non_selected_candidates':
            np.sqrt(np.mean(l_err_non_selected_candidates))
        }

    ld_err = []
    for num_obs in v_num_obs_test:
        print("Num. obs: ", num_obs)
        d_err = get_error(t_locs, m_meas, num_obs=num_obs, num_iter=4000)
        print(d_err)
        ld_err.append(d_err)

    def list_of_dicts_to_dict_of_lists(ld_in):
        d_out = {}
        for key in ld_in[0].keys():
            d_out[key] = [d[key] for d in ld_in]
        return d_out

    d_err = list_of_dicts_to_dict_of_lists(ld_err)

    G = GFigure(xlabel='Number of observations', ylabel='RMSE [dB]')
    #G.add_curve(v_num_obs_test, d_err['err_obs'], legend='RMSE[n]')
    G.add_curve(v_num_obs_test,
                d_err['err_non_selected_candidates'],
                legend='RMSE (random selection)')
    G.add_curve(v_num_obs_test,
                d_err['err_selected_candidate'],
                legend='RMSE (proposed)')

    return G


def get_rmse_curves(l_file_inds,
                    patch_side_length,
                    height,
                    folder,
                    num_points_x,
                    gridpoint_spacing,
                    evaluation_mode,
                    num_mc_iterations,
                    l_num_obs,
                    l_estimators,
                    b_insite_data=False):

    if b_insite_data:
        map_generator = InsiteMapGenerator(l_file_num=l_file_inds,
                                           patch_side_len=num_points_x *
                                           gridpoint_spacing,
                                           z_coord=height,
                                           folder=folder,
                                           gridpoint_spacing=gridpoint_spacing)
    else:
        map_generator = RealDataMapGenerator(
            l_file_num=l_file_inds,
            patch_side_len=patch_side_length,
            z_coord=height,
            folder=folder,
            gridpoint_spacing=gridpoint_spacing)

    grid = RectangularGrid(num_points_x=num_points_x,
                           num_points_y=num_points_x,
                           gridpoint_spacing=gridpoint_spacing,
                           height=height)

    for estimator in l_estimators:
        estimator.grid = grid

    l_mse = MapEstimationSimulator.compare_estimators_monte_carlo(
        map_generator=map_generator,
        num_mc_iterations=num_mc_iterations,
        l_estimators=l_estimators,
        evaluation_mode=evaluation_mode,
        l_num_obs=l_num_obs)

    F = MapEstimationSimulator.plot_rmse(
        l_mse,
        ylabel=f"RMSE({evaluation_mode}) [dB]",
        xlabel='Number of measurements')

    return F


def train_selected_model(model,
                         folder_in,
                         file_name,
                         batch_size,
                         validation_split,
                         b_load_weights_from=False,
                         load_weights_from=None,
                         save_weights_to=None,
                         **training_args):
    """
    This function trains the selected model and returns the history of the training.
    """

    def get_train_val_test_tf_dataset(folder_in, file_name, batch_size,
                                      validation_split):

        # load the data
        with open(folder_in + file_name, "rb") as f:
            d_train_and_test_data = pickle.load(f)

        # shuffle the train data
        x_train, y_train = sklearn.utils.shuffle(
            d_train_and_test_data["x_train"], d_train_and_test_data["y_train"])

        num_validation_samples = int(x_train.shape[0] * validation_split)

        # split the data into train and validation
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train[num_validation_samples:],
             y_train[num_validation_samples:])).shuffle(1000).batch(batch_size)

        val_ds = tf.data.Dataset.from_tensor_slices(
            (x_train[:num_validation_samples],
             y_train[:num_validation_samples])).batch(batch_size)

        # test data
        test_ds = tf.data.Dataset.from_tensor_slices(
            (d_train_and_test_data["x_test"],
             d_train_and_test_data["y_test"])).batch(batch_size)

        return train_ds, val_ds, test_ds

    assert save_weights_to is not None

    if b_load_weights_from:
        assert load_weights_from is not None

    train_ds, val_ds, test_ds = get_train_val_test_tf_dataset(
        folder_in, file_name, batch_size, validation_split)

    if b_load_weights_from:
        model.load_weights(load_weights_from)

    dict_history = model.train(train_dataset=train_ds,
                               validation_dataset=val_ds,
                               save_weights_to=save_weights_to,
                               test_dataset=test_ds,
                               **training_args)

    return dict_history


class ExperimentSet(gsim.AbstractExperimentSet):

    ############################################################################
    # 10. Dataset generation experiments
    ############################################################################

    # USRP, 32x32 grid, 6 feats, 120 meas, without maps 6, 17
    def experiment_1100(l_args):

        height = 5
        num_points_x = 32
        gridpoint_spacing = 1.2
        num_obs = 120
        num_feat = 6

        gen_dataset(map_generator=RealDataMapGenerator(
            l_file_num=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            patch_side_len=num_points_x * gridpoint_spacing,
            z_coord=height,
            folder=folder_datasets +
            'rme_datasets/usrp_data/grid_spacing_120_cm-freq_918_MHz/'),
                    save_to_folder=
                    f'output/datasets/usrp_{num_points_x}x{num_points_x}/',
                    num_obs=num_obs,
                    num_examples_per_map=10,
                    num_patches_to_gen=80,
                    num_feat=num_feat)

    # Gradiant, 16x16 grid, 6 feats, 100 meas, maps [0, 1]
    def experiment_1200(l_args):

        metric = 'rsrp'
        folder = folder_datasets + f'rme_datasets/gradiant/combined/{metric}/'
        l_file_inds_train = [0, 1]
        height = 5
        gridpoint_spacing = 4
        num_points_x = 16

        num_obs = 100
        num_feat = 6

        gen_dataset(
            map_generator=RealDataMapGenerator(
                l_file_num=l_file_inds_train,
                patch_side_len=num_points_x * gridpoint_spacing,
                z_coord=height,
                folder=folder,
                gridpoint_spacing=gridpoint_spacing),
            save_to_folder=
            f'output/datasets/gradiant_{num_points_x}x{num_points_x}_{metric}_train_800k/',
            num_obs=num_obs,
            num_examples_per_map=10,
            num_patches_to_gen=80000,
            num_feat=num_feat)

    # Ray-tracing, 16x16 grid, 6 feats, 100 meas, maps 1->40
    def experiment_1305(l_args):

        gridpoint_spacing = 4
        num_points_x = 16
        height = 20
        num_obs = 100
        num_feat = 6

        map_generator = InsiteMapGenerator(
            l_file_num=np.arange(1, 41),
            patch_side_len=num_points_x * gridpoint_spacing,
            z_coord=height,
            folder=folder_datasets + "insite_data/power_rosslyn/",
            gridpoint_spacing=gridpoint_spacing)

        gen_dataset(
            map_generator=map_generator,
            save_to_folder=
            f'output/datasets/ray_tracing_{num_points_x}x{num_points_x}_train_800k/',
            num_obs=num_obs,
            num_examples_per_map=10,
            num_patches_to_gen=80000,
            num_feat=num_feat)

    ############################################################################
    # 20. Estimator training experiments
    ############################################################################

    # USRP, transformers
    # 120 meas, 6 feats, 2 heads, 48 dim embedding, 4 layers
    # before fixing the bug in reshape_for_multihead_attention
    def experiment_2100(l_args):
        num_obs = 120  # + 1 for the target location
        num_feat = 6
        file_path = 'output/datasets/usrp_32x32_train_800k/transformer_dataset_meas-' \
            + str(num_obs) + '_feats-' + str(num_feat)

        dataset = load_tensor_dataset(file_path + '.pth')

        print(f'Num. train. examples: {len(dataset)}')
        print(f'Num. obs: {num_obs-1}')

        return train_transformer(
            dataset,
            dataset_val=None,
            nn_folder=os.path.join(
                dnn_folder,
                f"transformer_{inspect.currentframe().f_code.co_name}"),
            num_candidates_train=None,
            context_len_min=40,
            num_feat=num_feat,
            num_heads=2,
            dim_embedding=48,  # 48
            num_layers=4,  # 4
            dropout_prob=0.0,
            b_causal_masking=True,
            device_type='mps',
            num_epochs=10,
            val_split=0.2,
            lr=5e-5,
            batch_size=1024,
            batch_size_eval=1024,
            lr_patience=20,
            lr_decay=0.8,
            shuffle=True,
            first_epoch_to_plot=4,
            llc=LossLandscapeConfig(epoch_inds=[],
                                    max_num_directions=4,
                                    neg_gradient_step_scales=np.linspace(
                                        -2e-3, 2e-3, 19)))

    # USRP, transformers
    # 120 meas, 6 feats, 2 heads, 48 dim embedding, 4 layers
    # after fixing the bug in reshape_for_multihead_attention
    def experiment_2100_5(l_args):
        num_obs = 120  # + 1 for the target location
        num_feat = 6
        file_path = 'output/datasets/usrp_32x32_train_800k/transformer_dataset_meas-' \
            + str(num_obs) + '_feats-' + str(num_feat)

        dataset = load_tensor_dataset(file_path + '.pth')

        print(f'Num. train. examples: {len(dataset)}')
        print(f'Num. obs: {num_obs}')

        return train_transformer(
            dataset,
            dataset_val=None,
            nn_folder=os.path.join(
                dnn_folder,
                f"transformer_{inspect.currentframe().f_code.co_name}"),
            context_len_min=40,
            num_feat=num_feat,
            num_heads=2,
            dim_embedding=48,  # 48
            num_layers=4,  # 4
            dropout_prob=0.0,
            b_causal_masking=True,
            device_type='cuda:0',
            num_epochs=1000,
            val_split=0.2,
            lr=5e-5,
            batch_size=2048,
            batch_size_eval=1024,
            lr_patience=20,
            lr_decay=0.8,
            shuffle=True,
            first_epoch_to_plot=4,
            llc=LossLandscapeConfig(epoch_inds=[],
                                    max_num_directions=4,
                                    neg_gradient_step_scales=np.linspace(
                                        -2e-3, 2e-3, 19)))

    # USRP, transformers
    # 120 meas, 6 feats, 6 heads, 192 dim embedding, 6 layers
    # after fixing the bug in reshape_for_multihead_attention
    def experiment_2100_10(l_args):
        num_obs = 120  # + 1 for the target location
        num_feat = 6
        file_path = 'output/datasets/usrp_32x32_train_800k/transformer_dataset_meas-' \
            + str(num_obs) + '_feats-' + str(num_feat)

        dataset = load_tensor_dataset(file_path + '.pth')

        print(f'Num. train. examples: {len(dataset)}')
        print(f'Num. obs: {num_obs}')

        return train_transformer(
            dataset,
            dataset_val=None,
            nn_folder=os.path.join(
                dnn_folder,
                f"transformer_{inspect.currentframe().f_code.co_name}"),
            context_len_min=40,
            num_feat=num_feat,
            num_heads=6,
            dim_embedding=192,
            num_layers=6,
            dropout_prob=0.0,
            b_causal_masking=True,
            device_type='cuda:1',
            num_epochs=1000,
            val_split=0.2,
            lr=5e-5,
            batch_size=1024,
            batch_size_eval=1024,
            lr_patience=20,
            lr_decay=0.8,
            shuffle=True,
            first_epoch_to_plot=4,
            llc=LossLandscapeConfig(epoch_inds=[],
                                    max_num_directions=4,
                                    neg_gradient_step_scales=np.linspace(
                                        -2e-3, 2e-3, 19)))

    # USRP, transformers
    # 120 meas, 6 feats, 4 heads, 96 dim embedding, 8 layers
    # after fixing the bug in reshape_for_multihead_attention
    def experiment_2100_15(l_args):
        num_obs = 120  # + 1 for the target location
        num_feat = 6
        file_path = 'output/datasets/usrp_32x32_train_800k/transformer_dataset_meas-' \
            + str(num_obs) + '_feats-' + str(num_feat)

        dataset = load_tensor_dataset(file_path + '.pth')

        print(f'Num. train. examples: {len(dataset)}')
        print(f'Num. obs: {num_obs}')

        return train_transformer(
            dataset,
            dataset_val=None,
            nn_folder=os.path.join(
                dnn_folder,
                f"transformer_{inspect.currentframe().f_code.co_name}"),
            context_len_min=40,
            num_feat=num_feat,
            num_heads=4,
            dim_embedding=6 * 16,  # 48
            num_layers=8,  # 4
            dropout_prob=0.0,
            b_causal_masking=True,
            device_type='cuda:2',
            num_epochs=1000,
            val_split=0.2,
            lr=5e-5,
            batch_size=1024,
            batch_size_eval=1024,
            lr_patience=20,
            lr_decay=0.8,
            shuffle=True,
            first_epoch_to_plot=4,
            llc=LossLandscapeConfig(epoch_inds=[],
                                    max_num_directions=4,
                                    neg_gradient_step_scales=np.linspace(
                                        -2e-3, 2e-3, 19)))

    # Gradiant, transformers
    # 100 meas, 6 feats, 2 heads, 48 dim embedding, 4 layers
    # before fixing the bug in reshape_for_multihead_attention
    def experiment_2105(l_args):
        num_obs = 100  # + 1 for the target location
        num_feat = 6
        file_path = 'output/datasets/gradiant_16x16_rsrp_train_800k/transformer_dataset_meas-' \
            + str(num_obs) + '_feats-' + str(num_feat)

        dataset = load_tensor_dataset(file_path + '.pth')

        print(f'Num. train. examples: {len(dataset)}')
        print(f'Num. obs: {num_obs}')

        return train_transformer(
            dataset,
            dataset_val=None,
            nn_folder=os.path.join(
                dnn_folder,
                f"transformer_{inspect.currentframe().f_code.co_name}"),
            context_len_min=10,
            num_feat=num_feat,
            num_heads=2,
            dim_embedding=48,
            num_layers=4,
            dropout_prob=0.0,
            b_causal_masking=True,
            device_type='cuda:1',
            num_epochs=1000,
            val_split=0.2,
            lr=5e-6,
            batch_size=1024,
            batch_size_eval=1024,
            lr_patience=20,
            lr_decay=0.8,
            shuffle=True,
            first_epoch_to_plot=4,
            llc=LossLandscapeConfig(epoch_inds=[],
                                    max_num_directions=4,
                                    neg_gradient_step_scales=np.linspace(
                                        -2e-3, 2e-3, 19)))

    # Gradiant, transformers
    # 100 meas, 6 feats, 2 heads, 48 dim embedding, 4 layers
    # after fixing the bug in reshape_for_multihead_attention
    def experiment_2105_1(l_args):
        num_obs = 100  # + 1 for the target location
        num_feat = 6
        file_path = 'output/datasets/gradiant_16x16_rsrp_train_800k/transformer_dataset_meas-' \
            + str(num_obs) + '_feats-' + str(num_feat)

        dataset = load_tensor_dataset(file_path + '.pth')

        print(f'Num. train. examples: {len(dataset)}')
        print(f'Num. obs: {num_obs}')

        return train_transformer(
            dataset,
            dataset_val=None,
            nn_folder=os.path.join(
                dnn_folder,
                f"transformer_{inspect.currentframe().f_code.co_name}"),
            context_len_min=10,
            num_feat=num_feat,
            num_heads=2,
            dim_embedding=48,
            num_layers=4,
            dropout_prob=0.0,
            b_causal_masking=True,
            device_type='cuda:2',
            num_epochs=1000,
            val_split=0.2,
            lr=5e-5,
            batch_size=1024,
            batch_size_eval=1024,
            lr_patience=20,
            lr_decay=0.8,
            shuffle=True,
            first_epoch_to_plot=4,
            llc=LossLandscapeConfig(epoch_inds=[],
                                    max_num_directions=4,
                                    neg_gradient_step_scales=np.linspace(
                                        -2e-3, 2e-3, 19)))

    # Gradiant, transformers
    # 100 meas, 6 feats, 2 heads, 96 dim embedding, 6 layers
    # before fixing the bug in reshape_for_multihead_attention
    def experiment_2105_3(l_args):
        num_obs = 100  # + 1 for the target location
        num_feat = 6
        file_path = 'output/datasets/gradiant_16x16_rsrp_train_800k/transformer_dataset_meas-' \
            + str(num_obs) + '_feats-' + str(num_feat)

        dataset = load_tensor_dataset(file_path + '.pth')

        print(f'Num. train. examples: {len(dataset)}')
        print(f'Num. obs: {num_obs}')

        return train_transformer(
            dataset,
            dataset_val=None,
            nn_folder=os.path.join(
                dnn_folder,
                f"transformer_{inspect.currentframe().f_code.co_name}"),
            context_len_min=10,
            num_feat=num_feat,
            num_heads=2,
            dim_embedding=96,
            num_layers=6,
            dropout_prob=0.0,
            b_causal_masking=True,
            device_type='cuda:2',
            num_epochs=1000,
            val_split=0.2,
            lr=5e-5,
            batch_size=2048,
            batch_size_eval=1024,
            lr_patience=20,
            lr_decay=0.8,
            shuffle=True,
            first_epoch_to_plot=4,
            llc=LossLandscapeConfig(epoch_inds=[],
                                    max_num_directions=4,
                                    neg_gradient_step_scales=np.linspace(
                                        -2e-3, 2e-3, 19)))

    # Ray tracing, transformers
    # 100 meas, 6 feats, 2 heads, 48 dim embedding, 4 layers
    # after fixing the bug in reshape_for_multihead_attention
    def experiment_2110(l_args):
        num_obs = 100  # + 1 for the target location
        num_feat = 6
        file_path = 'output/datasets/ray_tracing_16x16_train_800k/transformer_dataset_meas-' \
            + str(num_obs) + '_feats-' + str(num_feat)

        dataset = load_tensor_dataset(file_path + '.pth')

        print(f'Num. train. examples: {len(dataset)}')
        print(f'Num. obs: {num_obs-1}')

        # question: should I choose different models for different datasets?

        return train_transformer(
            dataset,
            dataset_val=None,
            nn_folder=os.path.join(
                dnn_folder,
                f"transformer_{inspect.currentframe().f_code.co_name}"),
            context_len_min=10,
            num_feat=num_feat,
            num_heads=2,
            dim_embedding=48,
            num_layers=4,
            dropout_prob=0.0,
            b_causal_masking=True,
            device_type='cuda:3',
            num_epochs=1000,
            val_split=0.2,
            lr=2e-4,
            batch_size=2048,
            batch_size_eval=1024,
            lr_patience=20,
            lr_decay=0.8,
            shuffle=True,
            first_epoch_to_plot=4,
            llc=LossLandscapeConfig(epoch_inds=[],
                                    max_num_directions=4,
                                    neg_gradient_step_scales=np.linspace(
                                        -2e-3, 2e-3, 19)))

    # Ray tracing, KNN
    # 16x16 grid, 100 meas, 10->100 obs, height 20
    def experiment_2200(l_args):

        evaluation_mode_train = 'uniform_standard'
        dataset_name = 'ray_tracing'
        folder_out = 'output/trained_estimators/'

        gridpoint_spacing = 4
        num_points_x = 16
        height = 20
        num_mc_iter_train = 1000
        num_obs_train = (10, 100)

        l_estimators = [
            KNNEstimator(name_on_figs='KNN 1',
                         d_train_params=OrderedDict(
                             {'num_neighbors': np.arange(2, 13)})),
            # GudmundsonBatchKrigingEstimator(d_train_params=OrderedDict({
            #     'shadowing_std':
            #     np.arange(0.01, 1, 0.1),
            #     'shadowing_correlation_dist':
            #     np.array([50, 100, 150, 200, 250]),
            # })),
            # KernelRidgeRegressionEstimator(d_train_params=OrderedDict(
            #     {
            #         'kernel_width': 4 * np.arange(1, 11),
            #         'log_reg_par': np.arange(-12, 0),
            #         'kernel_type': ['gaussian', 'laplacian']
            #     })),
        ]

        for estimator in l_estimators:
            print(f'Training {estimator.__class__.__name__}')

            G = estimator.train(map_generator=InsiteMapGenerator(
                l_file_num=np.arange(1, 41),
                patch_side_len=num_points_x * gridpoint_spacing,
                z_coord=height,
                folder=folder_datasets + f'insite_data/power_rosslyn/',
                gridpoint_spacing=gridpoint_spacing),
                                num_obs=num_obs_train,
                                evaluation_mode=evaluation_mode_train,
                                num_mc_iter=num_mc_iter_train,
                                verbosity=5)

            # save the estimator
            folder_path = folder_out + f"{dataset_name}-{num_points_x}x{num_points_x}-{num_obs_train[0]}_to_{num_obs_train[1]}_meas/" + estimator.__class__.__name__ + "/"
            estimator.save_estimator(folder_path)

            # save the figure containing the training curves
            pickle.dump(G, open(folder_path + "G_train.pickle", "wb"))

        return

    # Ray tracing, Kriging
    # 16x16 grid, 100 meas, 10->100 obs, height 20
    def experiment_2200_3(l_args):

        evaluation_mode_train = 'uniform_standard'
        dataset_name = 'ray_tracing'
        folder_out = 'output/trained_estimators/'

        gridpoint_spacing = 4
        num_points_x = 16
        height = 20
        num_mc_iter_train = 1000
        num_obs_train = (10, 100)

        l_estimators = [
            # KNNEstimator(name_on_figs='KNN 1',
            #              d_train_params=OrderedDict(
            #                  {'num_neighbors': np.arange(2, 13)})),
            GudmundsonBatchKrigingEstimator(d_train_params=OrderedDict({
                'shadowing_std':
                np.arange(0.01, 1, 0.1),
                'shadowing_correlation_dist':
                np.array([50, 100, 150, 200, 250]),
            })),
            # KernelRidgeRegressionEstimator(d_train_params=OrderedDict(
            #     {
            #         'kernel_width': 4 * np.arange(1, 11),
            #         'log_reg_par': np.arange(-12, 0),
            #         'kernel_type': ['gaussian', 'laplacian']
            #     })),
        ]

        for estimator in l_estimators:

            print(f'Training {estimator.__class__.__name__}')

            G = estimator.train(map_generator=InsiteMapGenerator(
                l_file_num=np.arange(1, 41),
                patch_side_len=num_points_x * gridpoint_spacing,
                z_coord=height,
                folder=folder_datasets + f'insite_data/power_rosslyn/',
                gridpoint_spacing=gridpoint_spacing),
                                num_obs=num_obs_train,
                                evaluation_mode=evaluation_mode_train,
                                num_mc_iter=num_mc_iter_train,
                                verbosity=5)

            # save the estimator
            folder_path = folder_out + f"{dataset_name}-{num_points_x}x{num_points_x}-{num_obs_train[0]}_to_{num_obs_train[1]}_meas/" + estimator.__class__.__name__ + "/"
            estimator.save_estimator(folder_path)

            # save the figure containing the training curves
            pickle.dump(G, open(folder_path + "G_train.pickle", "wb"))

        return

    # Ray tracing, KRR
    # 16x16 grid, 100 meas, 10->100 obs, height 20
    def experiment_2200_7(l_args):

        evaluation_mode_train = 'uniform_standard'
        dataset_name = 'ray_tracing'
        folder_out = 'output/trained_estimators/'

        gridpoint_spacing = 4
        num_points_x = 16
        height = 20
        num_mc_iter_train = 1000
        num_obs_train = (10, 100)

        l_estimators = [
            # KNNEstimator(name_on_figs='KNN 1',
            #              d_train_params=OrderedDict(
            #                  {'num_neighbors': np.arange(2, 13)})),
            # GudmundsonBatchKrigingEstimator(d_train_params=OrderedDict({
            #     'shadowing_std':
            #     np.arange(0.01, 1, 0.1),
            #     'shadowing_correlation_dist':
            #     np.array([50, 100, 150, 200, 250]),
            # })),
            KernelRidgeRegressionEstimator(d_train_params=OrderedDict(
                {
                    'kernel_width': 4 * np.arange(1, 11),
                    'log_reg_par': np.arange(-12, 0),
                    'kernel_type': ['gaussian', 'laplacian']
                })),
        ]

        for estimator in l_estimators:

            print(f'Training {estimator.__class__.__name__}')

            G = estimator.train(map_generator=InsiteMapGenerator(
                l_file_num=np.arange(1, 41),
                patch_side_len=num_points_x * gridpoint_spacing,
                z_coord=height,
                folder=folder_datasets + f'insite_data/power_rosslyn/',
                gridpoint_spacing=gridpoint_spacing),
                                num_obs=num_obs_train,
                                evaluation_mode=evaluation_mode_train,
                                num_mc_iter=num_mc_iter_train,
                                verbosity=5)

            # save the estimator
            folder_path = folder_out + f"{dataset_name}-{num_points_x}x{num_points_x}-{num_obs_train[0]}_to_{num_obs_train[1]}_meas/" + estimator.__class__.__name__ + "/"
            estimator.save_estimator(folder_path)

            # save the figure containing the training curves
            pickle.dump(G, open(folder_path + "G_train.pickle", "wb"))

        return

    # Ray tracing
    # create training and testing datasets for DNN benchmarks
    # 16x16 grid, height 20
    def experiment_2400(l_args):
        # copied from 7201 in spectrum_measurement_experiments

        # ray-tracing dataset
        folder_in = folder_datasets + 'insite_data/power_rosslyn/'
        folder_out = 'output/datasets/ray_tracing_DNN/'
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)

        num_points_x = 16
        num_points_y = 16
        gridpoint_spacing = 4
        height = 20  # 5
        l_file_num_train = np.arange(1, 41)
        l_file_num_test = [41, 42]

        # parameters to get training and testing data
        num_maps_train = 30000  # 30000
        num_blocks_per_map_train = 5
        num_maps_test = 1000  # 1000
        num_blocks_per_map_test = 5
        num_meas_fraction_per_map = (10 / 256, 100 / 256
                                     )  # can be a float or a tuple of floats

        grid = RectangularGrid(area_side_length=None,
                               num_points_x=num_points_x,
                               num_points_y=num_points_y,
                               gridpoint_spacing=gridpoint_spacing,
                               height=height)

        # copied from get_dnn_train_test_dataset in
        # spectrum_measurement_experiments
        def get_dnn_data(num_maps, num_blocks_per_map,
                         num_meas_fraction_per_map, l_file_num,
                         l_aux_map_estimators):
            """
            Returns the data for the neural network.
            
            Args:
                num_maps: Number of maps to generate num_blocks_per_map: Number
                of blocks per map num_meas_fraction_per_map: Number of
                measurements per map as a fraction of the total number of grid
                points. It can be a float or a tuple of floats. If it is a
                tuple, then the number of measurements is chosen uniformly at
                random from the interval [num_meas_fraction_per_map[0],
                num_meas_fraction_per_map[1]].
            """

            map_generator = InsiteMapGenerator(
                l_file_num=l_file_num,
                patch_side_len=num_points_x * gridpoint_spacing,
                z_coord=height,
                folder=folder_in,
                gridpoint_spacing=gridpoint_spacing)

            x_data, y_data = NeuralNetworkEstimator.generate_dataset_for_nn(
                map_generator,
                grid,
                num_maps=num_maps,
                num_blocks_per_map=num_blocks_per_map,
                num_meas_fraction_per_map=num_meas_fraction_per_map,
                l_aux_map_estimators=l_aux_map_estimators)

            return x_data, y_data

        x_train, y_train = get_dnn_data(
            num_maps=num_maps_train,
            num_blocks_per_map=num_blocks_per_map_train,
            num_meas_fraction_per_map=num_meas_fraction_per_map,
            l_file_num=l_file_num_train,
            l_aux_map_estimators=None)

        x_test, y_test = get_dnn_data(
            num_maps=num_maps_test,
            num_blocks_per_map=num_blocks_per_map_test,
            num_meas_fraction_per_map=num_meas_fraction_per_map,
            l_file_num=l_file_num_test,
            l_aux_map_estimators=None)

        # save the data in dict as pickle
        d_train_and_test_data = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "l_file_num_train": l_file_num_train,
            "l_file_num_test": l_file_num_test,
        }

        with open(
                folder_out +
                f"num_maps_train_{num_maps_train}_grid_{num_points_x}x{num_points_y}_num_blocks_train_{num_blocks_per_map_train}_num_meas_frac_{num_meas_fraction_per_map}.pickle",
                "wb") as f:
            pickle.dump(d_train_and_test_data, f)

    # Ray tracing
    # train model 1: UnetStdAwareNnEstimator
    def experiment_2405_1(l_args):

        # copied from exp_7205 from spectrum_measurement_experiments

        # 1. Choose the data set
        folder_in = 'output/datasets/ray_tracing_DNN/'
        file_name = 'num_maps_train_30000_grid_16x16_num_blocks_train_5_num_meas_frac_(0.0390625, 0.390625).pickle'
        batch_size = 200
        validation_split = 0.2  # fraction of training data to be used for validation

        # 2. Choose the root folder for saving the weights
        weightfolder = "output/trained_estimators/ray-tracing/"
        if not os.path.exists(weightfolder):
            os.makedirs(weightfolder)

        # 3.  Choose the model
        ind_model = 1
        b_load_weights_from = True

        if ind_model == 1:
            model = UnetStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "unet_std_aware_estimator/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [5],
                'loss_metric': 'Custom',
                'l_learning_rate': [5e-5],
            }
        elif ind_model == 2:
            # this model is an Unet with Krr as the auxiliary map estimator
            model = UnetStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "unet_std_aware_estimator_with_KNN_GBK_KRR/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-3],
            }

        elif ind_model == 3:
            model = SurveyingStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "surveying_std_aware_estimator/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-5],
            }

        elif ind_model == 4:
            model = CompletionAutoencoderEstimator(
                height=16,
                width=16,
            )
            weight_subfolder = weightfolder + "completion_autoencoder/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        elif ind_model == 5:
            model = RadioUnetEstimator()
            weight_subfolder = weightfolder + "radio_unet/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        elif ind_model == 6:
            model = CompletionAutoencoderEstimator(
                height=16,
                width=16,
                n_channels=
                5  # 2 input data + 3 auxiliary map estimates (KNN, GBK, KRR)
            )
            weight_subfolder = weightfolder + "completion_autoencoder_with_KNN_GBK_KRR/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        else:
            raise ValueError('ind_model must be 1, 2, 3, 4 or 5')

        # 4. Train model
        dict_history = train_selected_model(model, folder_in, file_name,
                                            batch_size, validation_split,
                                            b_load_weights_from,
                                            load_weights_from, save_weights_to,
                                            **training_args)

        # 5. Plot the loss values
        G = model.plot_loss(dict_history)

        return G

    # Ray tracing
    # train model 2: UnetStdAwareNnEstimator
    def experiment_2405_2(l_args):

        # copied from exp_7205 from spectrum_measurement_experiments

        # 1. Choose the data set
        folder_in = 'output/datasets/ray_tracing_DNN/'
        file_name = 'num_maps_train_30000_grid_16x16_num_blocks_train_5_num_meas_frac_(0.0390625, 0.390625).pickle'
        batch_size = 200
        validation_split = 0.2  # fraction of training data to be used for validation

        # 2. Choose the root folder for saving the weights
        weightfolder = "output/trained_estimators/ray-tracing/"
        if not os.path.exists(weightfolder):
            os.makedirs(weightfolder)

        # 3.  Choose the model
        ind_model = 2
        b_load_weights_from = False

        if ind_model == 1:
            model = UnetStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "unet_std_aware_estimator/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-3],
            }
        elif ind_model == 2:
            # this model is an Unet with Krr as the auxiliary map estimator
            model = UnetStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "unet_std_aware_estimator_with_KNN_GBK_KRR/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-4],
            }

        elif ind_model == 3:
            model = SurveyingStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "surveying_std_aware_estimator/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-5],
            }

        elif ind_model == 4:
            model = CompletionAutoencoderEstimator(
                height=16,
                width=16,
            )
            weight_subfolder = weightfolder + "completion_autoencoder/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        elif ind_model == 5:
            model = RadioUnetEstimator()
            weight_subfolder = weightfolder + "radio_unet/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        elif ind_model == 6:
            model = CompletionAutoencoderEstimator(
                height=16,
                width=16,
                n_channels=
                5  # 2 input data + 3 auxiliary map estimates (KNN, GBK, KRR)
            )
            weight_subfolder = weightfolder + "completion_autoencoder_with_KNN_GBK_KRR/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        else:
            raise ValueError('ind_model must be 1, 2, 3, 4 or 5')

        # 4. Train model
        dict_history = train_selected_model(model, folder_in, file_name,
                                            batch_size, validation_split,
                                            b_load_weights_from,
                                            load_weights_from, save_weights_to,
                                            **training_args)

        # 5. Plot the loss values
        G = model.plot_loss(dict_history)

        return G

    # Ray tracing
    # train model 3: SurveyingStdAwareNnEstimator
    def experiment_2405_3(l_args):

        # copied from exp_7205 from spectrum_measurement_experiments

        # 1. Choose the data set
        folder_in = 'output/datasets/ray_tracing_DNN/'
        file_name = 'num_maps_train_30000_grid_16x16_num_blocks_train_5_num_meas_frac_(0.0390625, 0.390625).pickle'
        batch_size = 200
        validation_split = 0.2  # fraction of training data to be used for validation

        # 2. Choose the root folder for saving the weights
        weightfolder = "output/trained_estimators/ray-tracing/"
        if not os.path.exists(weightfolder):
            os.makedirs(weightfolder)

        # 3.  Choose the model
        ind_model = 3
        b_load_weights_from = True

        if ind_model == 1:
            model = UnetStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "unet_std_aware_estimator/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-3],
            }
        elif ind_model == 2:
            # this model is an Unet with Krr as the auxiliary map estimator
            model = UnetStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "unet_std_aware_estimator_with_KNN_GBK_KRR/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-3],
            }

        elif ind_model == 3:
            model = SurveyingStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "surveying_std_aware_estimator/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [5e-5],
            }

        elif ind_model == 4:
            model = CompletionAutoencoderEstimator(
                height=16,
                width=16,
            )
            weight_subfolder = weightfolder + "completion_autoencoder/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        elif ind_model == 5:
            model = RadioUnetEstimator()
            weight_subfolder = weightfolder + "radio_unet/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        elif ind_model == 6:
            model = CompletionAutoencoderEstimator(
                height=16,
                width=16,
                n_channels=
                5  # 2 input data + 3 auxiliary map estimates (KNN, GBK, KRR)
            )
            weight_subfolder = weightfolder + "completion_autoencoder_with_KNN_GBK_KRR/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        else:
            raise ValueError('ind_model must be 1, 2, 3, 4 or 5')

        # 4. Train model
        dict_history = train_selected_model(model, folder_in, file_name,
                                            batch_size, validation_split,
                                            b_load_weights_from,
                                            load_weights_from, save_weights_to,
                                            **training_args)

        # 5. Plot the loss values
        G = model.plot_loss(dict_history)

        return G

    # Ray tracing
    # train model 4: CompletionAutoencoderEstimator
    def experiment_2405_4(l_args):

        # copied from exp_7205 from spectrum_measurement_experiments

        # 1. Choose the data set
        folder_in = 'output/datasets/ray_tracing_DNN/'
        file_name = 'num_maps_train_30000_grid_16x16_num_blocks_train_5_num_meas_frac_(0.0390625, 0.390625).pickle'
        batch_size = 200
        validation_split = 0.2  # fraction of training data to be used for validation

        # 2. Choose the root folder for saving the weights
        weightfolder = "output/trained_estimators/ray-tracing/"
        if not os.path.exists(weightfolder):
            os.makedirs(weightfolder)

        # 3.  Choose the model
        ind_model = 4
        b_load_weights_from = True

        if ind_model == 1:
            model = UnetStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "unet_std_aware_estimator/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-4],
            }
        elif ind_model == 2:
            # this model is an Unet with Krr as the auxiliary map estimator
            model = UnetStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "unet_std_aware_estimator_with_KNN_GBK_KRR/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-4],
            }

        elif ind_model == 3:
            model = SurveyingStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "surveying_std_aware_estimator/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-4],
            }

        elif ind_model == 4:
            model = CompletionAutoencoderEstimator(
                height=16,
                width=16,
            )
            weight_subfolder = weightfolder + "completion_autoencoder/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 15,
                'learning_rate': 5e-5,
                'loss': "rmse",
            }

        elif ind_model == 5:
            model = RadioUnetEstimator()
            weight_subfolder = weightfolder + "radio_unet/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-4,
                'loss': "rmse",
            }

        elif ind_model == 6:
            model = CompletionAutoencoderEstimator(
                height=16,
                width=16,
                n_channels=
                5  # 2 input data + 3 auxiliary map estimates (KNN, GBK, KRR)
            )
            weight_subfolder = weightfolder + "completion_autoencoder_with_KNN_GBK_KRR/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        else:
            raise ValueError('ind_model must be 1, 2, 3, 4 or 5')

        # 4. Train model
        dict_history = train_selected_model(model, folder_in, file_name,
                                            batch_size, validation_split,
                                            b_load_weights_from,
                                            load_weights_from, save_weights_to,
                                            **training_args)

        # 5. Plot the loss values
        G = model.plot_loss(dict_history)

        return G

    # Ray tracing
    # train model 5: RadioUnetEstimator
    def experiment_2405_5(l_args):

        # copied from exp_7205 from spectrum_measurement_experiments

        # 1. Choose the data set
        folder_in = 'output/datasets/ray_tracing_DNN/'
        file_name = 'num_maps_train_30000_grid_16x16_num_blocks_train_5_num_meas_frac_(0.0390625, 0.390625).pickle'
        batch_size = 200
        validation_split = 0.2  # fraction of training data to be used for validation

        # 2. Choose the root folder for saving the weights
        weightfolder = "output/trained_estimators/ray-tracing/"
        if not os.path.exists(weightfolder):
            os.makedirs(weightfolder)

        # 3.  Choose the model
        ind_model = 5
        b_load_weights_from = True

        if ind_model == 1:
            model = UnetStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "unet_std_aware_estimator/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-3],
            }
        elif ind_model == 2:
            # this model is an Unet with Krr as the auxiliary map estimator
            model = UnetStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "unet_std_aware_estimator_with_KNN_GBK_KRR/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-3],
            }

        elif ind_model == 3:
            model = SurveyingStdAwareNnEstimator(
                sample_scaling=0.5,
                meas_output_same=False,
            )
            weight_subfolder = weightfolder + "surveying_std_aware_estimator/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'l_alpha': [0],
                'l_epochs': [3],
                'loss_metric': 'Custom',
                'l_learning_rate': [1e-5],
            }

        elif ind_model == 4:
            model = CompletionAutoencoderEstimator(
                height=16,
                width=16,
            )
            weight_subfolder = weightfolder + "completion_autoencoder/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        elif ind_model == 5:
            model = RadioUnetEstimator()
            weight_subfolder = weightfolder + "radio_unet/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 5,
                'learning_rate': 5e-5,
                'loss': "rmse",
            }

        elif ind_model == 6:
            model = CompletionAutoencoderEstimator(
                height=16,
                width=16,
                n_channels=
                5  # 2 input data + 3 auxiliary map estimates (KNN, GBK, KRR)
            )
            weight_subfolder = weightfolder + "completion_autoencoder_with_KNN_GBK_KRR/"
            load_weights_from = (weight_subfolder + "best_weight")
            save_weights_to = (weight_subfolder + "weights_")
            training_args = {
                'epochs': 3,
                'learning_rate': 1e-9,
                'loss': "rmse",
            }

        else:
            raise ValueError('ind_model must be 1, 2, 3, 4 or 5')

        # 4. Train model
        dict_history = train_selected_model(model, folder_in, file_name,
                                            batch_size, validation_split,
                                            b_load_weights_from,
                                            load_weights_from, save_weights_to,
                                            **training_args)

        # 5. Plot the loss values
        G = model.plot_loss(dict_history)

        return G

    # Radio mapper, transformers
    def experiment_2600(l_args):
        num_obs = 15  # + 1 for the target location
        num_feat = 6
        num_points_x = 32
        file_path = f'output/datasets/radio_mapper_{num_points_x}x{num_points_x}_{num_obs}_obs/' + 'preprocessed_transformer_dataset-' \
            + str(num_obs+1) + '_meas-' + str(num_feat) + '_feats'

        dataset = load_tensor_dataset(file_path + '.pth')

        print(f'Num. train. examples: {len(dataset)}')
        print(f'Num. obs: {num_obs-1}')

        # question: should I choose different models for different datasets?

        return train_transformer(
            dataset,
            dataset_val=None,
            nn_folder=os.path.join(
                dnn_folder,
                f"transformer_{inspect.currentframe().f_code.co_name}"),
            context_len_min=1,
            num_feat=num_feat,
            num_heads=2,
            dim_embedding=48,
            num_layers=4,
            dropout_prob=0.0,
            b_causal_masking=True,
            device_type='mps',
            num_epochs=10000,
            val_split=0.2,
            lr=2e-4,
            batch_size=200,
            batch_size_eval=100,
            lr_patience=6,
            lr_decay=0.8,
            shuffle=True,
            first_epoch_to_plot=4,
            llc=LossLandscapeConfig(epoch_inds=[],
                                    max_num_directions=4,
                                    neg_gradient_step_scales=np.linspace(
                                        -2e-3, 2e-3, 19)))

    ############################################################################
    # 31. Experiments to test and compare estimators
    ############################################################################

    # plot map estimates
    def experiment_3100(l_args):
        # copied from exp 4506 in spectrum_measurement_experiments

        # random_seed = np.random.randint(1000)
        random_seed = 894
        print(f'Random seed: {random_seed}')
        np.random.seed(random_seed)  #100

        folder = folder_datasets + 'rme_datasets/usrp_data/grid_spacing_120_cm-freq_918_MHz/'
        gridpoint_spacing = 1.2
        ind_file = 6  # 6 (obs-seed: 40-882; 60-425,564,823; 80-537,672; 100-418,537,682,777,925)
        # ind_file = 17
        estimator_path = folder_datasets + 'estimators/JPaper/usrp-32x32-40_to_150_meas/'
        num_points_x = 32

        patch_side_len = num_points_x * gridpoint_spacing

        sampling_mode = 'uniform_standard'
        num_obs = 110
        height = 5
        plotting_mode = 'imshow'  # in {'contour3D','imshow' , 'surface'}
        # zinterpolation = 'bilinear'
        zinterpolation = 'none'

        grid = RectangularGrid(num_points_x=num_points_x,
                               num_points_y=num_points_x,
                               gridpoint_spacing=gridpoint_spacing,
                               height=height)

        map_generator = RealDataMapGenerator(
            l_file_num=[ind_file],
            patch_side_len=patch_side_len,
            z_coord=height,
            folder=folder,
            gridpoint_spacing=gridpoint_spacing)

        num_feat = 6
        l_estimators = [
            KNNEstimator().load_estimator(estimator_path + "KNNEstimator/"),
            GudmundsonBatchKrigingEstimator().load_estimator(
                estimator_path + "GudmundsonBatchKrigingEstimator/"),
            KernelRidgeRegressionEstimator().load_estimator(
                estimator_path + "KernelRidgeRegressionEstimator/"),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 1',
                estimator=CompletionAutoencoderEstimator(
                    height=num_points_x,
                    width=num_points_x,
                    load_weights_from=estimator_path +
                    'completion_autoencoder/best_weight')),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 2',
                estimator=RadioUnetEstimator(load_weights_from=estimator_path +
                                             'radio_unet/best_weight')),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 3',
                estimator=UnetStdAwareNnEstimator(
                    load_weights_from=estimator_path +
                    'unet_std_aware_estimator/best_weight')),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 4',
                estimator=SurveyingStdAwareNnEstimator(
                    load_weights_from=estimator_path +
                    f'surveying_std_aware_estimator/' + 'best_weight')),
            AttnMapEstimator(
                num_feat=num_feat,
                att_dnn=TransformerDnn(
                    TransformerDnnConf(
                        dim_input=num_feat + 1,  # num_feat + 1
                        # max_context_len=num_obs - 1,
                        num_heads=2,
                        dim_embedding=48,
                        num_layers=4,
                        dropout_prob=0,
                        b_causal_masking=True,
                        device_type='mps',
                    ),
                    nn_folder=os.path.join(dnn_folder,
                                           "transformer_experiment_2100_5"),
                ))
        ]

        l_estimators[0].name_on_figs = 'K-NN'
        l_estimators[1].name_on_figs = 'Kriging'
        l_estimators[2].name_on_figs = 'KRR'
        l_estimators[-1].name_on_figs = 'STORM (Proposed)'
        # l_estimators[3].name_on_figs = 'Transformer'

        rmap = map_generator.generate_map()
        rmap_obs = rmap.get_obs(sampling_mode=sampling_mode, num_obs=num_obs)

        l_G_map_est = []

        for map_estimator in l_estimators:

            map_estimator.grid = grid  # Used for evaluating the map [consider changing this]

            map_estimate = map_estimator.estimate(
                rmap_obs)["t_power_map_estimate"].t_meas_gf
            # test_locs=rmap.m_meas_locs_sf)

            # rmap.grid = grid  # for plotting purposes
            rmap_est = Map(
                grid=grid,
                t_meas_gf=map_estimate  #d_map_estimate["t_power_map_estimate"]
            )

            l_G_map_est.append(
                rmap_est.plot(mode=plotting_mode,
                              m_meas_locs_sf=rmap_obs.m_meas_locs_sf))
            l_G_map_est[-1].l_subplots[0].title = map_estimator.name_on_figs
            l_G_map_est[-1].l_subplots[0].layout = "tight"
            l_G_map_est[-1].l_subplots[0].colorbar = False
            l_G_map_est[-1].l_subplots[0].ylabel = None
            l_G_map_est[-1].l_subplots[0].yticks = []
            l_G_map_est[-1].l_subplots[0].zinterpolation = zinterpolation
            # l_G_map_est[-1].l_subplots[0].zlim = (np.min(rmap.t_meas_gf),
            #                                       np.max(rmap.t_meas_gf))
        # True map
        G_map = rmap.plot(mode=plotting_mode,
                          m_meas_locs_sf=rmap_obs.m_meas_locs_sf)
        G_map.l_subplots[0].title = 'True Map'
        G_map.l_subplots[0].layout = ""
        G_map.l_subplots[0].colorbar = False
        G_map.l_subplots[0].zinterpolation = zinterpolation
        l_G_map_est[-1].l_subplots[0].zlim = (np.min(rmap.t_meas_gf),
                                              np.max(rmap.t_meas_gf))

        # put label of the color bar on GFIgure + tighter layout
        G = GFigure.concatenate([G_map] + l_G_map_est, num_subplot_columns=9)
        G.figsize = (25, 3)
        G.layout = 'tight'
        G.global_color_bar_position = [0.87, 0.24, 0.02, 0.545]
        G.global_color_bar_label = 'Power [dBm]'

        return G

    # plot a 3D map estimate
    def experiment_3100_3(l_args):

        random_seed = np.random.randint(1000)
        ind_map = np.random.randint(17)
        # random_seed = 894  # 894
        print(f'Random seed: {random_seed}')
        print(f'ind_map: {ind_map}')
        np.random.seed(random_seed)  #100

        folder = folder_datasets + 'rme_datasets/usrp_data/grid_spacing_120_cm-freq_918_MHz/'
        gridpoint_spacing = 1.2
        ind_file = ind_map  # 6 (obs-seed: 40-882; 60-425,564,823; 80-537,672; 100-418,537,682,777,925)
        # ind_file = 17
        num_points_x = 42
        height = 5

        # metric = 'rsrp'
        # folder = folder_datasets + f'rme_datasets/gradiant/combined/{metric}/'
        # l_file_inds = [2]
        # height = 5
        # gridpoint_spacing = 4
        # num_points_x = 16

        patch_side_len = num_points_x * gridpoint_spacing

        sampling_mode = 'uniform_standard'
        num_obs = 110

        grid = RectangularGrid(num_points_x=num_points_x,
                               num_points_y=num_points_x,
                               gridpoint_spacing=gridpoint_spacing,
                               height=height)

        estimator_path = folder_datasets + 'estimators/JPaper/usrp-32x32-40_to_150_meas/'
        estimator = KNNEstimator().load_estimator(estimator_path +
                                                  "KNNEstimator/")

        num_feat = 6
        # estimator = AttnMapEstimator(
        #     num_feat=num_feat,
        #     att_dnn=TransformerDnn(
        #         TransformerDnnConf(
        #             dim_input=num_feat + 1,  # num_feat + 1
        #             # max_context_len=num_obs - 1,
        #             num_heads=2,
        #             dim_embedding=48,
        #             num_layers=4,
        #             dropout_prob=0,
        #             b_causal_masking=True,
        #             device_type='mps',
        #         ),
        #         nn_folder=os.path.join(dnn_folder,
        #                                "transformer_experiment_2100_5"),
        #         # nn_folder=os.path.join(dnn_folder,
        #         #                        "transformer_experiment_2105"),
        #     ))
        l_G = []

        for ind_map in range(1, 18):

            map_generator = RealDataMapGenerator(
                l_file_num=ind_map,
                patch_side_len=patch_side_len,
                z_coord=height,
                folder=folder,
                gridpoint_spacing=gridpoint_spacing)

            rmap = map_generator.generate_map()
            rmap_obs = rmap.get_obs(sampling_mode=sampling_mode,
                                    num_obs=num_obs)

            estimator.grid = grid  # Used for evaluating the map [consider changing this]

            map_estimate = estimator.estimate(
                rmap_obs)["t_power_map_estimate"].t_meas_gf

            rmap_est = Map(
                grid=grid,
                t_meas_gf=map_estimate  #d_map_estimate["t_power_map_estimate"]
            )

            G = rmap_est.plot()

            G.global_color_bar_position = [.85, 0.25, 0.02, 0.5]

            # G.set_view_init(elev=29, azim=-149)

            import matplotlib.pyplot as plt

            F = G.plot()
            for ax in F.axes:
                if hasattr(ax, 'view_init'):
                    ax.view_init(elev=21, azim=-162)
            plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")
            plt.show()

            l_G.append(G)

        return l_G

    # USRP, knn, kriging, krr, tranformer
    # RMSE vs. num obs,
    # before fixing the bug in reshape_for_multihead_attention
    def experiment_3105(l_args):

        folder = folder_datasets + 'rme_datasets/usrp_data/grid_spacing_120_cm-freq_918_MHz/'
        l_file_inds = [6, 17]
        gridpoint_spacing = 1.2  # set to None for uniform sampling when generating the map
        num_points_x = 32
        patch_side_length = num_points_x * gridpoint_spacing

        height = 5
        num_mc_iterations = 1000  #00  # 1000
        l_num_obs = [40, 60, 80, 100, 120]
        # l_num_obs = [120]
        estimator_path = folder_datasets + 'estimators/JPaper/usrp-32x32-40_to_150_meas/'

        num_feat = 6
        #num_obs = 120

        l_estimators = [
            KNNEstimator().load_estimator(estimator_path + "KNNEstimator/"),
            GudmundsonBatchKrigingEstimator().load_estimator(
                estimator_path + "GudmundsonBatchKrigingEstimator/"),
            KernelRidgeRegressionEstimator().load_estimator(
                estimator_path + "KernelRidgeRegressionEstimator/"),
            AttnMapEstimator(
                num_feat=num_feat,
                att_dnn=TransformerDnn(
                    TransformerDnnConf(
                        dim_input=num_feat + 1,  # num_feat + 1
                        num_heads=2,
                        dim_embedding=48,
                        num_layers=4,
                        dropout_prob=0,
                        b_causal_masking=
                        True,  # It was trained with b_causal_masking=True
                        device_type='cuda:1',
                    ),
                    nn_folder=os.path.join(dnn_folder,
                                           "transformer_experiment_2100"),
                ))
        ]

        return get_rmse_curves(l_file_inds, patch_side_length, height, folder,
                               num_points_x, gridpoint_spacing,
                               'uniform_standard', num_mc_iterations,
                               l_num_obs, l_estimators)

    # USRP, transformer
    # RMSE vs. num obs,
    # after fixing the bug in reshape_for_multihead_attention
    def experiment_3105_1(l_args):

        folder = folder_datasets + 'rme_datasets/usrp_data/grid_spacing_120_cm-freq_918_MHz/'
        l_file_inds = [6, 17]
        gridpoint_spacing = 1.2  # set to None for uniform sampling when generating the map
        num_points_x = 32
        patch_side_length = num_points_x * gridpoint_spacing

        height = 5
        num_mc_iterations = 1000  #00  # 1000
        l_num_obs = [40, 60, 80, 100, 120]
        # l_num_obs = [120]
        estimator_path = folder_datasets + 'estimators/JPaper/usrp-32x32-40_to_150_meas/'

        num_feat = 6
        #num_obs = 120

        l_estimators = [
            # KNNEstimator().load_estimator(estimator_path + "KNNEstimator/"),
            # GudmundsonBatchKrigingEstimator().load_estimator(
            #     estimator_path + "GudmundsonBatchKrigingEstimator/"),
            # KernelRidgeRegressionEstimator().load_estimator(
            #     estimator_path + "KernelRidgeRegressionEstimator/"),
            AttnMapEstimator(
                num_feat=num_feat,
                att_dnn=TransformerDnn(
                    TransformerDnnConf(
                        dim_input=num_feat + 1,  # num_feat + 1
                        num_heads=2,
                        dim_embedding=48,
                        num_layers=4,
                        dropout_prob=0,
                        b_causal_masking=
                        True,  # It was trained with b_causal_masking=True
                        device_type='cuda:0',
                    ),
                    nn_folder=os.path.join(dnn_folder,
                                           "transformer_experiment_2100_5"),
                ))
        ]

        return get_rmse_curves(l_file_inds, patch_side_length, height, folder,
                               num_points_x, gridpoint_spacing,
                               'uniform_standard', num_mc_iterations,
                               l_num_obs, l_estimators)

    # USRP, tranformer
    # RMSE vs. num obs,
    # after fixing the bug in reshape_for_multihead_attention
    def experiment_3105_3(l_args):

        folder = folder_datasets + 'rme_datasets/usrp_data/grid_spacing_120_cm-freq_918_MHz/'
        l_file_inds = [6, 17]
        gridpoint_spacing = 1.2  # set to None for uniform sampling when generating the map
        num_points_x = 32
        patch_side_length = num_points_x * gridpoint_spacing

        height = 5
        num_mc_iterations = 1000  #00  # 1000
        l_num_obs = [40, 60, 80, 100, 120]  #
        # l_num_obs = [120]
        estimator_path = folder_datasets + 'estimators/JPaper/usrp-32x32-40_to_150_meas/'

        num_feat = 6
        #num_obs = 120

        l_estimators = [
            # KNNEstimator().load_estimator(estimator_path + "KNNEstimator/"),
            # GudmundsonBatchKrigingEstimator().load_estimator(
            #     estimator_path + "GudmundsonBatchKrigingEstimator/"),
            # KernelRidgeRegressionEstimator().load_estimator(
            #     estimator_path + "KernelRidgeRegressionEstimator/"),
            AttnMapEstimator(
                num_feat=num_feat,
                att_dnn=TransformerDnn(
                    TransformerDnnConf(
                        dim_input=num_feat + 1,  # num_feat + 1
                        num_heads=4,
                        dim_embedding=96,
                        num_layers=8,
                        dropout_prob=0,
                        b_causal_masking=
                        True,  # It was trained with b_causal_masking=True
                        device_type='cuda:1',
                    ),
                    nn_folder=os.path.join(dnn_folder,
                                           "transformer_experiment_2100_15"),
                ))
        ]

        return get_rmse_curves(l_file_inds, patch_side_length, height, folder,
                               num_points_x, gridpoint_spacing,
                               'uniform_standard', num_mc_iterations,
                               l_num_obs, l_estimators)

    # USRP, dnn benchmarks
    # RMSE vs. num obs
    def experiment_3105_4(l_args):

        folder = folder_datasets + 'rme_datasets/usrp_data/grid_spacing_120_cm-freq_918_MHz/'
        l_file_inds = [6, 17]
        gridpoint_spacing = 1.2  # set to None for uniform sampling when generating the map
        num_points_x = 32
        patch_side_length = num_points_x * gridpoint_spacing

        height = 5
        num_mc_iterations = 1000  #00  # 1000
        l_num_obs = [40, 60, 80, 100, 120]

        estimator_path = folder_datasets + 'estimators/JPaper/usrp-32x32-40_to_150_meas/'

        l_estimators = [
            # KernelRidgeRegressionEstimator().load_estimator(
            #     estimator_path + "KernelRidgeRegressionEstimator/"),
            # NeuralNetworkMapEstimator(
            #     name_on_figs='DNN 1',
            #     estimator=SurveyingStdAwareNnEstimator(
            #         load_weights_from=estimator_path +
            #         f'surveying_std_aware_estimator/' + 'best_weight')),
            # NeuralNetworkMapEstimator(
            #     name_on_figs='DNN 2',
            #     estimator=RadioUnetEstimator(load_weights_from=estimator_path +
            #                                  'radio_unet/best_weight')),
            # NeuralNetworkMapEstimator(
            #     name_on_figs='DNN 3',
            #     estimator=UnetStdAwareNnEstimator(
            #         load_weights_from=estimator_path +
            #         'unet_std_aware_estimator/best_weight')),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 4',
                estimator=CompletionAutoencoderEstimator(
                    height=num_points_x,
                    width=num_points_x,
                    load_weights_from=estimator_path +
                    'completion_autoencoder/best_weight'))
        ]

        return get_rmse_curves(l_file_inds, patch_side_length, height, folder,
                               num_points_x, gridpoint_spacing,
                               'uniform_standard', num_mc_iterations,
                               l_num_obs, l_estimators)

    # 4G Gradiant, knn, kriging, transformer
    # RMSE vs. num obs
    # before fixing the bug in reshape_for_multihead_attention
    def experiment_3107(l_args):

        folder = folder_datasets + 'rme_datasets/gradiant/combined/rsrp/'
        l_file_inds = [2]
        gridpoint_spacing = 4  # set to None for uniform sampling when generating the map
        num_points_x = 16
        patch_side_length = num_points_x * gridpoint_spacing

        height = 5
        num_mc_iterations = 1000  #00  # 1000
        l_num_obs = [40, 60, 80, 100, 120]

        estimator_path = folder_datasets + 'estimators/JPaper/gradiant-rsrp-16x16-10_to_100_meas/'

        num_feat = 6
        #num_obs = 120

        l_estimators = [
            KNNEstimator().load_estimator(estimator_path + "KNNEstimator/"),
            GudmundsonBatchKrigingEstimator().load_estimator(
                estimator_path + "GudmundsonBatchKrigingEstimator/"),
            # KernelRidgeRegressionEstimator().load_estimator(
            #     estimator_path + "KernelRidgeRegressionEstimator/"),
            AttnMapEstimator(
                num_feat=num_feat,
                att_dnn=TransformerDnn(
                    TransformerDnnConf(
                        dim_input=num_feat + 1,  # num_feat + 1
                        num_heads=2,
                        dim_embedding=48,
                        num_layers=4,
                        dropout_prob=0,
                        b_causal_masking=
                        True,  # It was trained with b_causal_masking=True
                        device_type='cuda:1',
                    ),
                    nn_folder=os.path.join(dnn_folder,
                                           "transformer_experiment_2100_15"),
                ))
        ]

        return get_rmse_curves(l_file_inds, patch_side_length, height, folder,
                               num_points_x, gridpoint_spacing,
                               'uniform_standard', num_mc_iterations,
                               l_num_obs, l_estimators)

    # 4G Gradiant, knn, kriging, transformer
    # RMSE vs. num obs
    # before fixing the bug in reshape_for_multihead_attention
    def experiment_3107(l_args):

        folder = folder_datasets + 'rme_datasets/gradiant/combined/rsrp/'
        l_file_inds = [2]
        gridpoint_spacing = 4  # set to None for uniform sampling when generating the map
        num_points_x = 16
        patch_side_length = num_points_x * gridpoint_spacing

        height = 5
        num_mc_iterations = 1000  #00  # 1000
        l_num_obs = [40, 60, 80, 100, 120]

        estimator_path = folder_datasets + 'estimators/JPaper/gradiant-rsrp-16x16-10_to_100_meas/'

        num_feat = 6
        #num_obs = 120

        l_estimators = [
            KNNEstimator().load_estimator(estimator_path + "KNNEstimator/"),
            GudmundsonBatchKrigingEstimator().load_estimator(
                estimator_path + "GudmundsonBatchKrigingEstimator/"),
            # KernelRidgeRegressionEstimator().load_estimator(
            #     estimator_path + "KernelRidgeRegressionEstimator/"),
            AttnMapEstimator(
                num_feat=num_feat,
                att_dnn=TransformerDnn(
                    TransformerDnnConf(
                        dim_input=num_feat + 1,  # num_feat + 1
                        num_heads=2,
                        dim_embedding=48,
                        num_layers=4,
                        dropout_prob=0,
                        masking=
                        True,  # It was trained with b_causal_masking=True
                        device_type='cuda:0',
                    ),
                    nn_folder=os.path.join(dnn_folder,
                                           "transformer_experiment_2105"),
                ))
        ]

        return get_rmse_curves(l_file_inds, patch_side_length, height, folder,
                               num_points_x, gridpoint_spacing,
                               'uniform_standard', num_mc_iterations,
                               l_num_obs, l_estimators)

    # 4G Gradiant, krr, dnn benchmarks
    # RMSE vs. num obs
    def experiment_3107_6(l_args):

        folder = folder_datasets + 'rme_datasets/gradiant/combined/rsrp/'
        l_file_inds = [2]
        gridpoint_spacing = 4  # set to None for uniform sampling when generating the map
        num_points_x = 16
        patch_side_length = num_points_x * gridpoint_spacing

        height = 5
        num_mc_iterations = 1000
        l_num_obs = [40, 60, 80, 100, 120]

        # estimator_path = folder_datasets + 'estimators/JPaper/usrp-32x32-40_to_150_meas/'
        estimator_path = folder_datasets + 'estimators/JPaper/gradiant-rsrp-16x16-10_to_100_meas/'

        l_estimators = [
            KernelRidgeRegressionEstimator().load_estimator(
                estimator_path + "KernelRidgeRegressionEstimator/"),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 1',
                estimator=SurveyingStdAwareNnEstimator(
                    load_weights_from=estimator_path +
                    f'surveying_std_aware_estimator/' + 'best_weight')),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 2',
                estimator=RadioUnetEstimator(load_weights_from=estimator_path +
                                             'radio_unet/best_weight')),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 3',
                estimator=UnetStdAwareNnEstimator(
                    load_weights_from=estimator_path +
                    'unet_std_aware_estimator/best_weight')),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 4',
                estimator=CompletionAutoencoderEstimator(
                    height=num_points_x,
                    width=num_points_x,
                    load_weights_from=estimator_path +
                    'completion_autoencoder/best_weight'))
        ]

        return get_rmse_curves(l_file_inds, patch_side_length, height, folder,
                               num_points_x, gridpoint_spacing,
                               'uniform_standard', num_mc_iterations,
                               l_num_obs, l_estimators)

    # Ray tracing, transformer
    # RMSE vs. num obs
    # after fixing the bug in reshape_for_multihead_attention
    def experiment_3112(l_args):

        folder = folder_datasets + 'insite_data/power_rosslyn/'
        l_file_inds = np.arange(41, 43)
        gridpoint_spacing = 4  # set to None for uniform sampling when generating the map
        num_points_x = 16
        patch_side_length = num_points_x * gridpoint_spacing

        height = 20
        num_mc_iterations = 1000  #00  # 1000
        l_num_obs = [20, 40, 60, 80, 100, 120]

        # estimator_path = folder_datasets + 'estimators/JPaper/usrp-32x32-40_to_150_meas/'
        estimator_path = folder_datasets + 'estimators/JPaper/gradiant-rsrp-16x16-10_to_100_meas/'

        num_feat = 6
        #num_obs = 120

        l_estimators = [
            KNNEstimator().load_estimator(estimator_path + "KNNEstimator/"),
            GudmundsonBatchKrigingEstimator().load_estimator(
                estimator_path + "GudmundsonBatchKrigingEstimator/"),
            KernelRidgeRegressionEstimator().load_estimator(
                estimator_path + "KernelRidgeRegressionEstimator/"),
            AttnMapEstimator(
                num_feat=num_feat,
                att_dnn=TransformerDnn(
                    TransformerDnnConf(
                        dim_input=num_feat + 1,  # num_feat + 1
                        num_heads=2,
                        dim_embedding=48,
                        num_layers=4,
                        dropout_prob=0,
                        b_causal_masking=
                        True,  # It was trained with b_causal_masking=True
                        device_type='cuda:3',
                    ),
                    nn_folder=os.path.join(dnn_folder,
                                           "transformer_experiment_2110"),
                ))
        ]

        return get_rmse_curves(l_file_inds,
                               patch_side_length,
                               height,
                               folder,
                               num_points_x,
                               gridpoint_spacing,
                               'uniform_standard',
                               num_mc_iterations,
                               l_num_obs,
                               l_estimators,
                               b_insite_data=True)

    # Ray tracing, knn, kriging, krr
    # RMSE vs. num obs
    # similar to 3112, when waiting for training the transformer
    def experiment_3112_5(l_args):

        folder = folder_datasets + 'insite_data/power_rosslyn/'
        l_file_inds = np.arange(41, 43)
        gridpoint_spacing = 4  # set to None for uniform sampling when generating the map
        num_points_x = 16
        patch_side_length = num_points_x * gridpoint_spacing

        height = 20
        num_mc_iterations = 100  #00  # 1000
        l_num_obs = [20, 40, 60, 80, 100, 120]

        # estimator_path = folder_datasets + 'estimators/JPaper/usrp-32x32-40_to_150_meas/'
        estimator_path = folder_datasets + 'estimators/JPaper/gradiant-rsrp-16x16-10_to_100_meas/'

        num_feat = 6
        #num_obs = 120

        l_estimators = [
            KNNEstimator().load_estimator(estimator_path + "KNNEstimator/"),
            GudmundsonBatchKrigingEstimator().load_estimator(
                estimator_path + "GudmundsonBatchKrigingEstimator/"),
            KernelRidgeRegressionEstimator().load_estimator(
                estimator_path + "KernelRidgeRegressionEstimator/"),
            # AttnMapEstimator(
            #     num_feat=num_feat,
            #     att_dnn=TransformerDnn(
            #         TransformerDnnConf(
            #             dim_input=num_feat + 1,  # num_feat + 1
            #             num_heads=2,
            #             dim_embedding=48,
            #             num_layers=4,
            #             dropout_prob=0,
            #             b_causal_masking=True,  # It was trained with b_causal_masking=True
            #             device_type='cuda:3',
            #         ),
            #         nn_folder=os.path.join(dnn_folder,
            #                                "transformer_experiment_2110"),
            #     ))
        ]

        return get_rmse_curves(l_file_inds,
                               patch_side_length,
                               height,
                               folder,
                               num_points_x,
                               gridpoint_spacing,
                               'uniform_standard',
                               num_mc_iterations,
                               l_num_obs,
                               l_estimators,
                               b_insite_data=True)

    # Ray tracing, other dnn benchmarks
    # RMSE vs. num obs
    def experiment_3112_10(l_args):

        folder = folder_datasets + 'insite_data/power_rosslyn/'
        l_file_inds = np.arange(41, 43)
        gridpoint_spacing = 4  # set to None for uniform sampling when generating the map
        num_points_x = 16
        patch_side_length = num_points_x * gridpoint_spacing

        height = 20
        num_mc_iterations = 1000  #00  # 1000
        l_num_obs = [20, 40, 60, 80, 100, 120]

        estimator_path = 'output/trained_estimators/ray-tracing/'

        l_estimators = [
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 1',
                estimator=SurveyingStdAwareNnEstimator(
                    load_weights_from=estimator_path +
                    f'surveying_std_aware_estimator/' + 'best_weight')),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 2',
                estimator=RadioUnetEstimator(load_weights_from=estimator_path +
                                             'radio_unet/best_weight')),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 3',
                estimator=UnetStdAwareNnEstimator(
                    load_weights_from=estimator_path +
                    'unet_std_aware_estimator/best_weight')),
            NeuralNetworkMapEstimator(
                name_on_figs='DNN 4',
                estimator=CompletionAutoencoderEstimator(
                    height=num_points_x,
                    width=num_points_x,
                    load_weights_from=estimator_path +
                    'completion_autoencoder/best_weight'))
        ]

        return get_rmse_curves(l_file_inds,
                               patch_side_length,
                               height,
                               folder,
                               num_points_x,
                               gridpoint_spacing,
                               'uniform_standard',
                               num_mc_iterations,
                               l_num_obs,
                               l_estimators,
                               b_insite_data=True)

    # USRP, knn, kriging
    # plot RMSE curves vs. different map indices
    def experiment_3115(l_args):

        folder = folder_datasets + 'rme_datasets/usrp_data/grid_spacing_120_cm-freq_918_MHz/'
        l_file_inds = list(range(18))

        gridpoint_spacing = 1.2  # set to None for uniform sampling when generating the map
        num_points_x = 32
        patch_side_length = num_points_x * gridpoint_spacing

        evaluation_mode = 'uniform_standard'

        height = 5
        num_mc_iterations = 200  # 1000
        l_num_obs = [50]

        estimator_path = folder_datasets + 'estimators/JPaper/usrp-32x32-40_to_150_meas/'
        l_estimators = [
            KNNEstimator().load_estimator(estimator_path + "KNNEstimator/"),
            GudmundsonBatchKrigingEstimator().load_estimator(
                estimator_path + "GudmundsonBatchKrigingEstimator/"),
        ]
        grid = RectangularGrid(num_points_x=num_points_x,
                               num_points_y=num_points_x,
                               gridpoint_spacing=gridpoint_spacing,
                               height=height)
        for estimator in l_estimators:
            estimator.grid = grid

        ll_rmse = [[], []]
        l_legend = []
        l_style = []

        for ind_file, ind_map in enumerate(l_file_inds):

            print(f'\nInd_map: {ind_map}')
            map_generator = RealDataMapGenerator(
                l_file_num=[ind_map],
                patch_side_len=patch_side_length,
                z_coord=height,
                folder=folder,
                gridpoint_spacing=gridpoint_spacing)

            l_mse = MapEstimationSimulator.compare_estimators_monte_carlo(
                map_generator=map_generator,
                num_mc_iterations=num_mc_iterations,
                l_estimators=l_estimators,
                evaluation_mode=evaluation_mode,
                l_num_obs=l_num_obs)

            for ind_estimator, (str_legend, v_x, v_y,
                                style) in enumerate(l_mse):

                ll_rmse[ind_estimator].append(np.sqrt(v_y[0]))

                if ind_file == 0:
                    l_legend.append(str_legend)
                    l_style.append(style)

        G = GFigure(xlabel='Map index', ylabel=f"RMSE({evaluation_mode}) [dB]")

        for ind_estimator, (str_legend,
                            style) in enumerate(zip(l_legend, l_style)):
            G.add_curve(l_file_inds,
                        ll_rmse[ind_estimator],
                        legend=str_legend,
                        styles=style)

        return G

    # combine RMSE at many points that were run separately
    def experiment_3120(l_args):

        path_to_results = 'output/transformer_experiments/experiment_'

        l_experiment_inds = [
            '2004_51_39',
            '2004_51_79',
            '2004_51_119',
            '2004_51_149',
        ]

        l_x = []
        ll_y = []
        for ind, experiment_ind in enumerate(l_experiment_inds):

            with open(path_to_results + experiment_ind + '.pk', 'rb') as f:
                G = pickle.load(f)[0]
            l_y = []
            for curve in G.l_subplots[0].l_curves:
                l_y += curve.yaxis

            l_x += G.l_subplots[0].l_curves[0].xaxis
            ll_y.append(l_y)

        v_x = np.array(l_x)
        m_y = np.array(ll_y)
        xlabel = G.l_subplots[0].xlabel
        ylabel = G.l_subplots[0].ylabel

        l_legend_str = [
            G.l_subplots[0].l_curves[ind].legend_str
            for ind in range(len(G.l_subplots[0].l_curves))
        ]

        l_styles = [
            G.l_subplots[0].l_curves[ind].style
            for ind in range(len(G.l_subplots[0].l_curves))
        ]

        G = GFigure(xlabel=xlabel, ylabel=ylabel)
        for ind, legend_str in enumerate(l_legend_str):
            G.add_curve(v_x,
                        m_y[:, ind],
                        legend=legend_str,
                        styles=l_styles[ind])

        return G

    # plot the RMSE curves for different transformer models
    # the models in this experiment were trained with masking [40, 120]
    def experiment_3125(l_args):

        folder = folder_datasets + 'rme_datasets/usrp_data/grid_spacing_120_cm-freq_918_MHz/'
        l_file_inds = [6, 17]
        gridpoint_spacing = 1.2  # set to None for uniform sampling when generating the map
        num_points_x = 32
        patch_side_length = num_points_x * gridpoint_spacing

        l_evaluation_modes = [
            'uniform_standard',
        ]
        height = 5
        num_mc_iterations = 1
        l_num_obs = [40, 60, 80, 100, 120]

        device_type = 'mps'

        l_estimators = [
            AttnMapEstimator(num_feat=5,
                             att_dnn=TransformerDnn(
                                 TransformerDnnConf(
                                     dim_input=5 + 1,
                                     num_heads=2,
                                     dim_embedding=48,
                                     num_layers=4,
                                     dropout_prob=0.0,
                                     masking=True,
                                     device_type=device_type,
                                 ),
                                 nn_folder=os.path.join(
                                     dnn_folder,
                                     "transformer_experiment_1503_5"),
                             )),
            AttnMapEstimator(num_feat=8,
                             att_dnn=TransformerDnn(
                                 TransformerDnnConf(
                                     dim_input=8 + 1,
                                     num_heads=2,
                                     dim_embedding=48,
                                     num_layers=4,
                                     dropout_prob=0.05,
                                     masking=True,
                                     device_type=device_type,
                                 ),
                                 nn_folder=os.path.join(
                                     dnn_folder,
                                     "transformer_experiment_1503_8"),
                             )),
            AttnMapEstimator(num_feat=5,
                             att_dnn=TransformerDnn(
                                 TransformerDnnConf(
                                     dim_input=5 + 1,
                                     num_heads=2,
                                     dim_embedding=48,
                                     num_layers=8,
                                     dropout_prob=0.05,
                                     masking=True,
                                     device_type=device_type,
                                 ),
                                 nn_folder=os.path.join(
                                     dnn_folder,
                                     "transformer_experiment_1505_5"),
                             )),
            AttnMapEstimator(num_feat=8,
                             att_dnn=TransformerDnn(
                                 TransformerDnnConf(
                                     dim_input=8 + 1,
                                     num_heads=2,
                                     dim_embedding=48,
                                     num_layers=8,
                                     dropout_prob=0.05,
                                     masking=True,
                                     device_type=device_type,
                                 ),
                                 nn_folder=os.path.join(
                                     dnn_folder,
                                     "transformer_experiment_1505_8"),
                             )),
            AttnMapEstimator(
                num_feat=8,
                att_dnn=TransformerDnn(
                    TransformerDnnConf(
                        dim_input=8 + 1,  # num_feat + 1
                        num_heads=3,
                        dim_embedding=96,
                        num_layers=6,
                        dropout_prob=0.0,
                        masking=True,
                        device_type=device_type,
                    ),
                    nn_folder=os.path.join(dnn_folder,
                                           "transformer_experiment_1507"),
                )),
            AttnMapEstimator(
                num_feat=8,
                att_dnn=TransformerDnn(
                    TransformerDnnConf(
                        dim_input=8 + 1,  # num_feat + 1
                        num_heads=4,
                        dim_embedding=192,
                        num_layers=12,
                        dropout_prob=0.05,
                        masking=True,
                        device_type=device_type,
                    ),
                    nn_folder=os.path.join(dnn_folder,
                                           "transformer_experiment_1510"),
                ))
        ]
        l_G = []
        for evaluation_mode in tqdm(l_evaluation_modes):
            G = get_rmse_curves(l_file_inds, patch_side_length, height, folder,
                                num_points_x, gridpoint_spacing,
                                evaluation_mode, num_mc_iterations, l_num_obs,
                                l_estimators)
            l_G.append(G)
        return l_G

    ############################################################################
    # 33. Custom experiments to combined results from previous experiments
    ############################################################################

    # USRP
    # after fixing the bug in reshape_for_multihead_attention
    def experiment_3300(l_args):
        path_to_results = 'output/transformer_experiments/experiment_'

        l_experiment_inds = [
            '3105',  # transformer before debugging reshape_for_multihead_attention
            '3105_4',  # dnn benchmarks
        ]

        l_y = []
        l_legend_str = []
        l_styles = []
        for ind, experiment_ind in enumerate(l_experiment_inds):

            with open(path_to_results + experiment_ind + '.pk', 'rb') as f:
                G = pickle.load(f)[0]

            for ind, curve in enumerate(G.l_subplots[0].l_curves):
                l_y.append(curve.yaxis)
                l_legend_str.append(curve.legend_str)
                l_styles.append(curve.style)

        l_x = G.l_subplots[0].l_curves[0].xaxis

        l_order = [0, 1, 3, 7, 5, 6, 4, 2]

        l_styles = [
            '.-#1f77b4', 'v-#2ca02c', 's-#d62728', '^-#ff7f0e', 'p-#9467bd',
            '*-#8c564b', 'X-#e377c2', 'P-#7f7f7f', 'D-#bcbd22', '.-#17becf'
        ]

        l_legend_str[l_legend_str.index('KNN 1')] = 'KNN'
        l_legend_str[l_legend_str.index('KrigingEstimator')] = 'Kriging'
        l_legend_str[l_legend_str.index('KernelRidgeRegression')] = 'KRR'
        l_legend_str[l_legend_str.index(
            'AttnMapEstimator')] = 'STORM (Proposed)'
        l_legend_str[l_legend_str.index('DNN 1')] = 'DNN 1 tempt'
        l_legend_str[l_legend_str.index('DNN 4')] = 'DNN 1'
        l_legend_str[l_legend_str.index('DNN 1 tempt')] = 'DNN 4'

        v_x = np.array(l_x)
        m_y = np.array(l_y)
        xlabel = 'Number of observations'  # G.l_subplots[0].xlabel
        ylabel = 'RMSE [dB]'  # G.l_subplots[0].ylabel

        G = GFigure(xlabel=xlabel, ylabel=ylabel)
        for ind in l_order:
            G.add_curve(v_x,
                        m_y[ind],
                        legend=l_legend_str[ind],
                        styles=l_styles[ind])

        G.plot()

        return G

    # Gradiant
    # before fixing the bug in reshape_for_multihead_attention
    def experiment_3307(l_args):
        path_to_results = 'output/transformer_experiments/experiment_'

        l_experiment_inds = [
            '3107',  # transformer before debugging reshape_for_multihead_attention
            '3107_6',  # dnn benchmarks
        ]

        l_y = []
        l_legend_str = []
        l_styles = []
        for ind, experiment_ind in enumerate(l_experiment_inds):

            with open(path_to_results + experiment_ind + '.pk', 'rb') as f:
                G = pickle.load(f)[0]

            for ind, curve in enumerate(G.l_subplots[0].l_curves):
                l_y.append(curve.yaxis)
                l_legend_str.append(curve.legend_str)
                l_styles.append(curve.style)

        l_x = G.l_subplots[0].l_curves[0].xaxis

        l_order = [0, 1, 3, 7, 5, 6, 4, 2]
        l_styles = [
            '.-#1f77b4', 'v-#2ca02c', 's-#d62728', '^-#ff7f0e', 'p-#9467bd',
            '*-#8c564b', 'X-#e377c2', 'P-#7f7f7f', 'D-#bcbd22', '.-#17becf'
        ]
        l_legend_str[l_legend_str.index('KNN 1')] = 'KNN'
        l_legend_str[l_legend_str.index('KrigingEstimator')] = 'Kriging'
        l_legend_str[l_legend_str.index('KernelRidgeRegression')] = 'KRR'
        l_legend_str[l_legend_str.index(
            'AttnMapEstimator')] = 'STORM (Proposed)'

        l_legend_str[l_legend_str.index('DNN 1')] = 'DNN 1 tempt'
        l_legend_str[l_legend_str.index('DNN 4')] = 'DNN 1'
        l_legend_str[l_legend_str.index('DNN 1 tempt')] = 'DNN 4'

        v_x = np.array(l_x)
        m_y = np.array(l_y)
        xlabel = 'Number of observations'  # G.l_subplots[0].xlabel
        ylabel = 'RMSE [dB]'  # G.l_subplots[0].ylabel

        G = GFigure(xlabel=xlabel, ylabel=ylabel)
        for ind in l_order:
            G.add_curve(v_x,
                        m_y[ind],
                        legend=l_legend_str[ind],
                        styles=l_styles[ind])

        G.plot()

        return G

    # Ray tracing
    # after fixing the bug in reshape_for_multihead_attention
    def experiment_3315(l_args):
        path_to_results = 'output/transformer_experiments/experiment_'

        l_experiment_inds = [
            '3112',
            '3112_10',
        ]

        l_y = []
        l_legend_str = []
        l_styles = []
        for ind, experiment_ind in enumerate(l_experiment_inds):

            with open(path_to_results + experiment_ind + '.pk', 'rb') as f:
                G = pickle.load(f)[0]

            for ind, curve in enumerate(G.l_subplots[0].l_curves):
                l_y.append(curve.yaxis)
                l_legend_str.append(curve.legend_str)
                l_styles.append(curve.style)

        l_x = G.l_subplots[0].l_curves[0].xaxis

        l_order = [0, 1, 2, 7, 5, 6, 3]  # 4
        l_styles = [
            '.-#1f77b4', 'v-#2ca02c', '^-#ff7f0e', 's-#d62728', 'p-#9467bd',
            '*-#8c564b', 'X-#e377c2', 'P-#7f7f7f', 'D-#bcbd22', '.-#17becf'
        ]
        l_legend_str[l_legend_str.index('KNN 1')] = 'KNN'
        l_legend_str[l_legend_str.index('KrigingEstimator')] = 'Kriging'
        l_legend_str[l_legend_str.index('KernelRidgeRegression')] = 'KRR'
        l_legend_str[l_legend_str.index(
            'AttnMapEstimator')] = 'STORM (Proposed)'
        l_legend_str[l_legend_str.index('DNN 1')] = 'DNN 1 tempt'
        l_legend_str[l_legend_str.index('DNN 4')] = 'DNN 1'
        l_legend_str[l_legend_str.index('DNN 1 tempt')] = 'DNN 4'

        v_x = np.array(l_x)
        m_y = np.array(l_y)
        xlabel = 'Number of observations'  # G.l_subplots[0].xlabel
        ylabel = 'RMSE [dB]'  # G.l_subplots[0].ylabel

        G = GFigure(xlabel=xlabel, ylabel=ylabel)
        for ind in l_order:
            G.add_curve(v_x,
                        m_y[ind],
                        legend=l_legend_str[ind],
                        styles=l_styles[ind])

        G.plot()

        return G

    ############################################################################
    # 40. Experiments for active sensing
    ############################################################################

    # USRP dataset
    def experiment_4000(l_args):

        np.random.seed(101)  #100
        torch.manual_seed(101)
        num_meas = 101  # + 1 for the target location
        num_feat = 6
        v_num_obs_test = np.array([4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98])
        v_num_obs_train = np.array([4, 70, 80, 90])

        num_points_x = 32
        gridpoint_spacing = 1.2
        num_examples_per_patch = 100
        num_patches_train = 50
        num_patches_test = 50
        dataset = 'usrp'

        map_generator, folder_train_dataset, folder_test_dataset = init_map_generator(
            patch_side_len=num_points_x * gridpoint_spacing,
            num_examples_per_patch=num_examples_per_patch,
            num_patches_train=num_patches_train,
            num_patches_test=num_patches_test,
            dataset=dataset)

        ade = AttnMapEstimator(
            num_feat=num_feat,
            att_dnn=TransformerDnn(
                TransformerDnnConf(
                    dim_input=num_feat + 1,  # num_feat + 1                
                    num_heads=2,
                    dim_embedding=20,
                    num_layers=4,
                    dropout_prob=0,
                    b_causal_masking=True,
                    device_type=None,
                ),
                nn_folder=os.path.join(
                    dnn_folder,
                    f"transformer_{inspect.currentframe().f_code.co_name}/high_num_obs"
                ),
            ))

        l_G = []

        # generate the training dataset
        b_gen_train_dataset = True
        if b_gen_train_dataset:
            gen_dataset(map_generator=map_generator,
                        save_to_folder=folder_train_dataset,
                        num_obs=num_meas - 1,
                        num_examples_per_map=num_examples_per_patch,
                        num_patches_to_gen=num_patches_train,
                        num_feat=num_feat)

        # train the models
        b_train = True
        if b_train:
            file_path = folder_train_dataset + get_preprocessed_file_base_name(
                num_meas, num_feat)

            dataset = load_tensor_dataset(file_path)

            print(f'Num. train. examples: {len(dataset)}')
            print(f'Num. meas: {num_meas-1}')

            G_train_ade = NeuralNet.plot_training_history(
                ade.train(
                    min_context_len=3,
                    num_candidates_train=num_meas - 1 - v_num_obs_train,
                    val_split=.2,
                    dataset_train=dataset,
                    dataset_val=None,
                    lr=1e-3,  #5e-5,
                    batch_size=20,
                    batch_size_eval=10,
                    lr_patience=20,
                    lr_decay=0.8,
                    patience=60,
                    shuffle=True,
                    num_epochs=10,
                    best_weights=True,
                ),
                first_epoch_to_plot=4)

            l_G += G_train_ade

        # Generate the test dataset
        b_gen_test_dataset = True
        if b_gen_test_dataset:
            gen_dataset(map_generator=map_generator,
                        save_to_folder=folder_test_dataset,
                        num_obs=num_meas - 1,
                        num_examples_per_map=num_examples_per_patch,
                        num_patches_to_gen=num_patches_test,
                        preprocess=False)

        b_test = True
        if b_test:
            G = test_transformers_on_active_sensing(
                num_meas=num_meas,
                attn_estimator=ade,
                v_num_obs_test=v_num_obs_test,
                folder_test_dataset=folder_test_dataset)
            l_G.append(G)

        return l_G

    # Ray-tracing dataset
    def experiment_4010(l_args):

        np.random.seed(101)  #100
        torch.manual_seed(101)
        num_meas = 101  # + 1 for the target location
        num_feat = 6
        v_num_obs_test = np.array([4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98])
        v_num_obs_train = np.array([4, 70, 80, 90])

        patch_side_len = 150
        num_examples_per_patch = 100
        num_patches_train = 10
        num_patches_test = 10
        dataset = 'ray-tracing'

        map_generator, folder_train_dataset, folder_test_dataset = init_map_generator(
            patch_side_len=patch_side_len,
            num_examples_per_patch=num_examples_per_patch,
            num_patches_train=num_patches_train,
            num_patches_test=num_patches_test,
            dataset=dataset)

        ade = AttnMapEstimator(
            num_feat=num_feat,
            att_dnn=TransformerDnn(
                TransformerDnnConf(
                    dim_input=num_feat + 1,  # num_feat + 1                
                    num_heads=2,
                    dim_embedding=20,
                    num_layers=4,
                    dropout_prob=0,
                    b_causal_masking=True,
                    device_type=None,
                ),
                nn_folder=os.path.join(
                    dnn_folder,
                    f"transformer_{inspect.currentframe().f_code.co_name}/medium_model"
                ),
            ))

        l_G = []

        # generate the training dataset
        b_gen_train_dataset = True
        if b_gen_train_dataset:
            gen_dataset(map_generator=map_generator,
                        save_to_folder=folder_train_dataset,
                        num_obs=num_meas - 1,
                        num_examples_per_map=num_examples_per_patch,
                        num_patches_to_gen=num_patches_train,
                        num_feat=num_feat)

        # train the models
        b_train = True
        if b_train:
            file_path = folder_train_dataset + get_preprocessed_file_base_name(
                num_meas, num_feat)

            dataset = load_tensor_dataset(file_path)

            print(f'Num. train. examples: {len(dataset)}')
            print(f'Num. meas: {num_meas-1}')

            G_train_ade = NeuralNet.plot_training_history(
                ade.train(
                    min_context_len=3,
                    num_candidates_train=num_meas - 1 - v_num_obs_train,
                    val_split=.2,
                    dataset_train=dataset,
                    dataset_val=None,
                    lr=1e-3,  #5e-5,
                    batch_size=20,
                    batch_size_eval=10,
                    lr_patience=5,
                    lr_decay=0.5,
                    patience=150,
                    shuffle=True,
                    num_epochs=2,
                    best_weights=True,
                ),
                first_epoch_to_plot=4)

            l_G += G_train_ade

        # Generate the test dataset
        b_gen_test_dataset = True
        if b_gen_test_dataset:
            gen_dataset(map_generator=map_generator,
                        save_to_folder=folder_test_dataset,
                        num_obs=num_meas - 1,
                        num_examples_per_map=num_examples_per_patch,
                        num_patches_to_gen=num_patches_test,
                        preprocess=False)

        b_test = True
        if b_test:

            G = test_transformers_on_active_sensing(
                num_meas=num_meas,
                attn_estimator=ade,
                v_num_obs_test=v_num_obs_test,
                folder_test_dataset=folder_test_dataset)
            l_G.append(G)

        return l_G
