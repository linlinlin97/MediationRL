import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from scipy.special import expit
from evaluator_Linear import evaluator
from Simulator import Simulator
from Policy import target_policy, control_policy
from probLearner import PMLearner, RewardLearner, PALearner
from ratioLearner import RatioLinearLearner as RatioLearner
from qLearner_Linear import Qlearner
from p_tqdm import p_imap

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    filename='triple_robust.log')
if __name__ == "__main__":
    CORES = 25
    dim_state = 2
    dim_mediator = 2
    std_M = 1
    std_S = 1
    simulator = Simulator(model_type='Gaussian_semi_s2m2',
                          dim_state=dim_state,
                          dim_mediator=dim_mediator,
                          std_M=std_M,
                          std_S=std_S)

    problearner_parameters = {
        "splitter": ["best", "random"],
        "max_depth": range(1, 40)
    }

    # truth
    true_DE, true_ME, true_SE = 0, 0, 0

    def toy_robust(num_trajectory,
                   num_time,
                   expectation_MCMC_iter=100,
                   expectation_MCMC_iter_Q3=100,
                   expectation_MCMC_iter_Q_diff=100,
                   ratio_ndim=5,
                   truncate=20,
                   seed=0):
        simulator.sample_trajectory(num_trajectory, num_time, seed)
        simulator.trajectory2iid()
        sim_iid_dataset = simulator.iid_dataset
        # print(sim_iid_dataset["next_state"].shape)
        # exit()
        est_obj1 = evaluator(sim_iid_dataset,
                             num_trajectory,
                             num_time,
                             Qlearner,
                             PMLearner,
                             RewardLearner,
                             PALearner,
                             RatioLearner,
                             problearner_parameters=problearner_parameters,
                             ratio_ndim=ratio_ndim,
                             truncate=truncate,
                             l2penalty=10**(-4),
                             target_policy=target_policy,
                             control_policy=control_policy,
                             dim_state=dim_state,
                             dim_mediator=2,
                             Q_settings={
                                 'scaler': 'Identity',
                                 'product_tensor': False,
                                 'beta': 2 / 7,
                                 'include_intercept': False,
                                 'expectation_MCMC_iter_Q3':
                                 expectation_MCMC_iter_Q3,
                                 'expectation_MCMC_iter_Q_diff':
                                 expectation_MCMC_iter_Q_diff,
                                 'penalty': 10**(-4),
                                 'd': 3,
                                 'min_L': 7,
                                 "t_dependent_Q": False
                             },
                             expectation_MCMC_iter=expectation_MCMC_iter,
                             seed=0)

        est_obj1.estimate_DE_ME_SE()
        est_value1 = est_obj1.est_DEMESE
        se_value1 = est_obj1.se_DEMESE
        IDE_MR = est_value1[0]
        IME_MR = est_value1[1]
        DDE_MR = est_value1[2]
        DME_MR = est_value1[3]

        IDE_Direct = est_value1[4]
        IME_Direct = est_value1[5]
        DDE_Direct = est_value1[6]
        DME_Direct = est_value1[7]

        IDE_WIS1 = est_value1[8]
        IME_WIS1 = est_value1[9]
        DDE_WIS1 = est_value1[10]
        DME_WIS1 = est_value1[11]

        IDE_WIS2 = est_value1[12]
        IME_WIS2 = est_value1[13]
        DDE_WIS2 = np.nan
        DME_WIS2 = np.nan

        IDE_baseline = est_value1[14]
        IME_baseline = est_value1[15]
        DDE_baseline = np.nan
        DME_baseline = np.nan

        return [
            IDE_MR, IME_MR, DDE_MR, DME_MR, IDE_Direct, IME_Direct, DDE_Direct,
            DME_Direct, IDE_WIS1, IME_WIS1, DDE_WIS1, DME_WIS1, IDE_WIS2,
            IME_WIS2, DDE_WIS2, DME_WIS2, IDE_baseline, IME_baseline,
            DDE_baseline, DME_baseline, num_trajectory, num_time, seed,
            est_obj1.eta_pie, est_obj1.eta_a0
        ]

    def toy_robust_star(args):
        return toy_robust(*args)

    def MC_toy_robust(nrep, num_trajectory, num_time, expectation_MCMC_iter,
                      expectation_MCMC_iter_Q3, expectation_MCMC_iter_Q_diff,
                      ratio_ndim, truncate):
        seed_list = np.arange(nrep, dtype='int64').tolist()
        num_trajectory_list = (np.ones(nrep, dtype='int64') *
                               num_trajectory).tolist()
        max_time_list = (np.ones(nrep, dtype='int64') * num_time).tolist()
        MCMC_iter_list = (np.ones(nrep, dtype='int64') *
                          expectation_MCMC_iter).tolist()
        MCMC_iter_Q3_list = (np.ones(nrep, dtype='int64') *
                             expectation_MCMC_iter_Q3).tolist()
        MCMC_iter_Q_diff_list = (np.ones(nrep, dtype='int64') *
                                 expectation_MCMC_iter_Q_diff).tolist()
        ratio_ndim_list = (np.ones(nrep, dtype='int64') * ratio_ndim).tolist()
        truncatef_list = (np.ones(nrep, dtype='int64') * truncate).tolist()
        param_list = zip(num_trajectory_list, max_time_list, MCMC_iter_list,
                         MCMC_iter_Q3_list, MCMC_iter_Q_diff_list,
                         ratio_ndim_list, truncatef_list, seed_list)

        # with Pool(CORES) as p:
        #     Error = list(tqdm(p.imap(toy_robust_star, param_list), total=nrep))
        Error = list(
            p_imap(toy_robust_star, param_list, num_cpus=CORES, total=nrep))
        return Error

    nrep = 25
    Error = dict()
    for num_time in np.arange(1, 2) * 100:
        Error[num_time] = dict()
        for num_trajectory in np.arange(1, 6) * 300:
            # for num_trajectory in np.arange(1, 3) * 200:
            Error[num_time][num_trajectory] = MC_toy_robust(
                nrep, num_trajectory, num_time, 20, 20, 20, 15, 500)
            with open(
                    "s2m2_estimate_additive_identity_L7_stdM_" + str(std_M) +
                    "_stdS_" + str(std_S) + '.txt', "wb") as fp:
                pickle.dump(Error, fp)
