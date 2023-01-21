import numpy as np
from scipy.special import expit
import sys, tqdm, warnings, itertools, pickle, time
import multiprocessing as mp
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

sys.path.append("../")
from evaluator_Linear import evaluator
from probLearner import PMLearner, RewardLearner, PALearner
from ratioLearner import RatioLinearLearner as RatioLearner
from qLearner_Linear import Qlearner
from IHS2018_Data.create_IHS2018_dataset import create_IHS2018_dataset_weekly, binary_split_dataset
from qbehavior import Learn_Behavior_Q
from _util import *

# HYPER PARAMETERS
SCALER = ["Standardize"]

# RUN SETUP
CORES = 2
SPECIALTY = ["all"]
SEED = 42


# TRT_STEPS = list(range(15, 25))
def make_one_run(specialty, t_dependent_Q, scaler, seed):
    t0 = time.time()
    dataset = create_IHS2018_dataset_weekly(specialty=specialty,
                                            d=1,
                                            start_week=0)
    T = dataset["T"]
    N = dataset["N"]
    d1, d2 = binary_split_dataset(dataset=dataset, seed=seed)
    est_obj1 = make_one_split(d1, d2, t_dependent_Q, scaler, seed)
    est_obj2 = make_one_split(d2, d1, t_dependent_Q, scaler, seed)
    ind_est_pooled = np.concatenate([est_obj1.ind_est, est_obj2.ind_est],
                                    axis=1)
    est_DEMESE = np.mean(ind_est_pooled, 1)
    if N > 100:
        se_DEMESE = np.array([
            np.mean(ind_est_pooled[:, i * T:(i + 1) * T], 1) for i in range(N)
        ]).std(0) / np.sqrt(N)
    else:
        se_DEMESE = np.std(ind_est_pooled, 1) / np.sqrt(N)

    t1 = time.time()
    return {
        "specialty": specialty,
        "t_dependent_Q": t_dependent_Q,
        "scaler": scaler,
        "est_value": est_DEMESE,
        "se_value": se_DEMESE,
        "time": "{:.0f}".format(t1 - t0)
    }


def make_one_split(dataset1, dataset2, t_dependent_Q, scaler, seed):
    sdim = dataset1["state"].shape[-1]
    mdim = dataset1["mediator"].shape[-1]

    # LEARN OPTIMAL POLICY
    problearner_parameters = {
        "splitter": ["best", "random"],
        "max_depth": range(1, 20)
    },
    Q_settings = {
        'scaler': 'Identity',
        'product_tensor': False,
        'beta': 3 / 7,
        'include_intercept': False,
        'expectation_MCMC_iter_Q3': 50,
        'expectation_MCMC_iter_Q_diff': 50,
        'penalty': 10**(-4),
        'd': 3,
        'min_L': 7
    }
    Q_behavior = Learn_Behavior_Q(dataset1, PMLearner, PALearner, sdim, mdim,
                                  problearner_parameters, Q_settings, seed)

    def optimal_policy(state=None, dim_state=1, action=None):
        opt_A = Q_behavior.opt_A(state)
        if action is None:
            action_value = opt_A
        else:
            action = np.copy(action).flatten()
            action_value = 1 - abs(opt_A - action)
        return action_value

    # def control_policy(state=None, dim_state=None, action=None, get_a=False):
    #     # fixed policy with fixed action 0
    #     if get_a:
    #         action_value = np.array([0])
    #     else:
    #         state = np.copy(state).reshape(-1, dim_state)
    #         NT = state.shape[0]
    #         if action is None:
    #             action_value = np.array([0] * NT)
    #         else:
    #             action = np.copy(action).flatten()
    #             if len(action) == 1 and NT > 1:
    #                 action = action * np.ones(NT)
    #             action_value = 1 - action
    #     return action_value

    def control_policy(state=None, dim_state=None, action=None, get_a=False):
        # fixed policy with fixed action 0
        if get_a:
            action_value = np.array([1])
        else:
            state = np.copy(state).reshape(-1, dim_state)
            NT = state.shape[0]
            if action is None:
                action_value = np.array([1] * NT)
            else:
                action = np.copy(action).flatten()
                if len(action) == 1 and NT > 1:
                    action = action * np.ones(NT)
                action_value = action
        return action_value

    # HYPER PARAMETERS
    #Fixed hyper-parameter--no need to modify
    expectation_MCMC_iter = 50
    expectation_MCMC_iter_Q3 = expectation_MCMC_iter_Q_diff = 50
    truncate = 50
    problearner_parameters = {
        "splitter": ["best", "random"],
        "max_depth": range(1, 50)
    },
    dim_state = 1
    dim_mediator = 2
    ratio_ndim = 10
    d = 3
    L = 10
    t_depend_target = False
    target_policy = optimal_policy
    control_policy = control_policy
    t_dependent_Q = t_dependent_Q
    scaler = scaler
    num_trajectory = dataset2["N"]
    num_time = dataset2["T"]

    est_obj1 = evaluator(dataset2,
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
                         t_depend_target=t_depend_target,
                         target_policy=target_policy,
                         control_policy=control_policy,
                         dim_state=dim_state,
                         dim_mediator=dim_mediator,
                         Q_settings={
                             'scaler': scaler,
                             'product_tensor': False,
                             'beta': 3 / 7,
                             'include_intercept': False,
                             'expectation_MCMC_iter_Q3':
                             expectation_MCMC_iter_Q3,
                             'expectation_MCMC_iter_Q_diff':
                             expectation_MCMC_iter_Q_diff,
                             'penalty': 10**(-4),
                             'd': d,
                             'min_L': L,
                             "t_dependent_Q": t_dependent_Q
                         },
                         expectation_MCMC_iter=expectation_MCMC_iter,
                         seed=10)

    est_obj1.estimate_DE_ME_SE()
    est_value1 = est_obj1.est_DEMESE
    se_value1 = est_obj1.se_DEMESE

    return est_obj1


def make_one_run_star(args):
    return make_one_run(*args)


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


if __name__ == "__main__":
    note = sys.argv[1]
    settings = expand_grid({
        "specialty": SPECIALTY,
        "t_dependent_Q": [False, True],
        # "t_dependent_Q": [False],
        "scaler": SCALER,
        "seed": [SEED]
    })

    all_jobs = []
    for index, row in settings.iterrows():
        all_jobs.append([
            row["specialty"], row["t_dependent_Q"], row["scaler"], row["seed"]
        ])

    with mp.Pool(CORES) as pool:
        output = list(
            tqdm.tqdm(pool.imap(make_one_run_star, all_jobs),
                      total=len(all_jobs),
                      desc="OPTIMAL"))

    pd.DataFrame(output).to_csv("./outs/optimal_cv_results_{}_{}.txt".format(
        note,
        datetime.now().strftime("%y%m%d%H%M%S")),
                                index=False,
                                sep="\t")
