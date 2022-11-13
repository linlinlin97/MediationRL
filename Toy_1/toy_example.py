import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from scipy.special import expit
from evaluator import evaluator
from Simulator import Simulator
from Policy import target_policy, control_policy
from utilize import true_q_function, false_q_function, true_ratio_function, false_ratio_function, true_pm_function, false_pm_function, true_reward_function, false_reward_function, true_pa_function, false_pa_function, true_eta_values, false_eta_values
from utilize_prototypemodel import QModel, RatioModel, PMModel, RewardModel, PAModel, EtaModel

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(message)s', filename='triple_robust.log')

simulator = Simulator(model_type='toy', dim_state=1)

#true model
q1_model0 = QModel(true_q_function, typ = "Q1")
q2_model0 = QModel(true_q_function, typ = "Q2")
q3_model0 = QModel(true_q_function, typ = "Q3")
q4_model0 = QModel(true_q_function, typ = "Q4")
ratio_model0 = RatioModel(true_ratio_function)
pm_model0 = PMModel(true_pm_function)
reward_model0 = RewardModel(true_reward_function)
pa_model0 = PAModel(true_pa_function)
eta_model0 = EtaModel(true_eta_values)


# truth
true_DE,true_ME,true_SE = -1.27668596, -1.2224647699999998, -3.06693133

def toy_robust(num_trajectory, num_time, seed):
    #false model
    q1_model1 = QModel(false_q_function, typ = "Q1", seed = seed)
    q2_model1 = QModel(false_q_function, typ = "Q2", seed = seed)
    q3_model1 = QModel(false_q_function, typ = "Q3", seed = seed)
    q4_model1 = QModel(false_q_function, typ = "Q4", seed = seed)
    pm_model1 = PMModel(false_pm_function, seed = seed)
    eta_model1 = EtaModel(false_eta_values, seed = seed)
    reward_model1 = RewardModel(false_reward_function, seed = seed)
    pa_model1 = PAModel(false_pa_function, seed = seed)
    ratio_model1 = RatioModel(false_ratio_function, seed = seed)

    simulator.sample_trajectory(num_trajectory, num_time, seed)
    simulator.trajectory2iid()
    sim_iid_dataset = simulator.iid_dataset
    
    est_obj0 = evaluator(sim_iid_dataset, q1_model0, q2_model0, q3_model0, q4_model0, ratio_model0, pm_model0,
                           reward_model0, pa_model0, eta_model0, target_policy = target_policy, control_policy = control_policy)
     
    est_obj0.estimate_DE_ME_SE()
    est_value0 = est_obj0.est_DEMESE
    var_value0 = est_obj0.var_DEMESE
    DE_error_correct0 = est_value0[0] - true_DE
    ME_error_correct0 = est_value0[1] - true_ME
    SE_error_correct0 = est_value0[2] - true_SE
    DE_var_correct0 = var_value0[0] 
    ME_var_correct0 = var_value0[1] 
    SE_var_correct0 = var_value0[2]
    
    est_obj1 = evaluator(sim_iid_dataset, q1_model1, q2_model1, q3_model1, q4_model1, ratio_model0, pm_model1,
                           reward_model0, pa_model0, eta_model1, target_policy = target_policy, control_policy = control_policy)
    
    est_obj1.estimate_DE_ME_SE()
    est_value1 = est_obj1.est_DEMESE
    var_value1 = est_obj1.var_DEMESE
    DE_error_correct1 = est_value1[0] - true_DE
    ME_error_correct1 = est_value1[1] - true_ME
    SE_error_correct1 = est_value1[2] - true_SE
    DE_var_correct1 = var_value1[0] 
    ME_var_correct1 = var_value1[1] 
    SE_var_correct1 = var_value1[2]
    
    est_obj2 = evaluator(sim_iid_dataset, q1_model1, q2_model1, q3_model1, q4_model1, ratio_model0, pm_model0,
                           reward_model1, pa_model0, eta_model1, target_policy = target_policy, control_policy = control_policy)
    
    est_obj2.estimate_DE_ME_SE()
    est_value2 = est_obj2.est_DEMESE
    var_value2 = est_obj2.var_DEMESE
    DE_error_correct2 = est_value2[0] - true_DE
    ME_error_correct2 = est_value2[1] - true_ME
    SE_error_correct2 = est_value2[2] - true_SE
    DE_var_correct2 = var_value2[0] 
    ME_var_correct2 = var_value2[1] 
    SE_var_correct2 = var_value2[2]
    
    est_obj3 = evaluator(sim_iid_dataset, q1_model0, q2_model0, q3_model0, q4_model0, ratio_model1, pm_model0,
                           reward_model0, pa_model1, eta_model0, target_policy = target_policy, control_policy = control_policy)
     
        
    est_obj3.estimate_DE_ME_SE()
    est_value3 = est_obj3.est_DEMESE
    var_value3 = est_obj3.var_DEMESE
    DE_error_correct3 = est_value3[0] - true_DE
    ME_error_correct3 = est_value3[1] - true_ME
    SE_error_correct3 = est_value3[2] - true_SE
    DE_var_correct3 = var_value3[0] 
    ME_var_correct3 = var_value3[1] 
    SE_var_correct3 = var_value3[2]
    
    
    est_obj4 = evaluator(sim_iid_dataset, q1_model1, q2_model1, q3_model1, q4_model1, ratio_model1, pm_model1,
                           reward_model1, pa_model1, eta_model1, target_policy = target_policy, control_policy = control_policy)
     
        
    est_obj4.estimate_DE_ME_SE()
    est_value4 = est_obj4.est_DEMESE
    var_value4 = est_obj4.var_DEMESE
    DE_error_correct4 = est_value4[0] - true_DE
    ME_error_correct4 = est_value4[1] - true_ME
    SE_error_correct4 = est_value4[2] - true_SE
    DE_var_correct4 = var_value4[0] 
    ME_var_correct4 = var_value4[1] 
    SE_var_correct4 = var_value4[2]
    
    
    return [num_trajectory, num_time, seed,
            DE_error_correct0,ME_error_correct0,SE_error_correct0,
            DE_error_correct1,ME_error_correct1,SE_error_correct1,
            DE_error_correct2,ME_error_correct2,SE_error_correct2,
            DE_error_correct3,ME_error_correct3,SE_error_correct3,
            DE_error_correct4,ME_error_correct4,SE_error_correct4,
            DE_var_correct0,ME_var_correct0,SE_var_correct0, 
            DE_var_correct1,ME_var_correct1,SE_var_correct1, 
            DE_var_correct2,ME_var_correct2,SE_var_correct2, 
            DE_var_correct3,ME_var_correct3,SE_var_correct3,
           DE_var_correct4,ME_var_correct4,SE_var_correct4]

def toy_robust_star(args):
    return toy_robust(*args)


def MC_toy_robust(nrep, num_trajectory, num_time):
    seed_list = np.arange(nrep, dtype='int64').tolist()
    num_trajectory_list = (np.ones(nrep, dtype='int64') * num_trajectory).tolist()
    max_time_list = (np.ones(nrep, dtype='int64') * num_time).tolist()
    param_list = zip(num_trajectory_list, max_time_list, seed_list)

    with Pool() as p:
        Error = list(tqdm(p.imap(toy_robust_star, param_list), total=nrep))
    return Error

num_time = 50
nrep = 200

Error = dict()
Error[num_time] = dict()
#for num_trajectory in np.arange(1, 11) * 100:
for num_trajectory in np.arange(1, 11) * 30:
    Error[num_time][num_trajectory] = MC_toy_robust(nrep, num_trajectory, num_time)
    
    with open("robust_final.txt","wb") as fp:
        pickle.dump(Error, fp)
