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
q5_model0 = QModel(true_q_function, typ = "Q5")
ratio_model0 = RatioModel(true_ratio_function)
pm_model0 = PMModel(true_pm_function)
reward_model0 = RewardModel(true_reward_function)
pa_model0 = PAModel(true_pa_function)
eta_model0 = EtaModel(true_eta_values)


# truth
true_IDE,true_IME,true_DDE, true_DME = -1.27668596, -1.2224647699999998, -2.98224644, -0.08468489000000012

def toy_robust(num_trajectory, num_time, seed):
    #false model
    q1_model1 = QModel(false_q_function, typ = "Q1", seed = seed)
    q2_model1 = QModel(false_q_function, typ = "Q2", seed = seed)
    q3_model1 = QModel(false_q_function, typ = "Q3", seed = seed)
    q4_model1 = QModel(false_q_function, typ = "Q4", seed = seed)
    q5_model1 = QModel(false_q_function, typ = "Q5", seed = seed)
    pm_model1 = PMModel(false_pm_function, seed = seed)
    eta_model1 = EtaModel(false_eta_values, seed = seed)
    reward_model1 = RewardModel(false_reward_function, seed = seed)
    pa_model1 = PAModel(false_pa_function, seed = seed)
    ratio_model1 = RatioModel(false_ratio_function, seed = seed)

    simulator.sample_trajectory(num_trajectory, num_time, seed)
    simulator.trajectory2iid()
    sim_iid_dataset = simulator.iid_dataset
    
    ## all true
    est_obj0 = evaluator(sim_iid_dataset, q1_model0, q2_model0, q3_model0, q4_model0, q5_model0, ratio_model0, pm_model0,
                           reward_model0, pa_model0, eta_model0, target_policy = target_policy, control_policy = control_policy)
     
    est_obj0.estimate_DE_ME_SE()
    est_value0 = est_obj0.est_DEMESE
    var_value0 = est_obj0.var_DEMESE
    IDE_error_correct0 = est_value0[0] - true_IDE
    IME_error_correct0 = est_value0[1] - true_IME
    DDE_error_correct0 = est_value0[2] - true_DDE
    DME_error_correct0 = est_value0[3] - true_DME
    IDE_var_correct0 = var_value0[0] 
    IME_var_correct0 = var_value0[1] 
    DDE_var_correct0 = var_value0[2]
    DME_var_correct0 = var_value0[3]
    
    ## ratio, reward, pa true -- M1
    est_obj1 = evaluator(sim_iid_dataset, q1_model1, q2_model1, q3_model1, q4_model1, q5_model1, ratio_model0, pm_model1,
                           reward_model0, pa_model0, eta_model1, target_policy = target_policy, control_policy = control_policy)
    
    est_obj1.estimate_DE_ME_SE()
    est_value1 = est_obj1.est_DEMESE
    var_value1 = est_obj1.var_DEMESE
    IDE_error_correct1 = est_value1[0] - true_IDE
    IME_error_correct1 = est_value1[1] - true_IME
    DDE_error_correct1 = est_value1[2] - true_DDE
    DME_error_correct1 = est_value1[3] - true_DME
    IDE_var_correct1 = var_value1[0] 
    IME_var_correct1 = var_value1[1] 
    DDE_var_correct1 = var_value1[2]
    DME_var_correct1 = var_value1[3]
    
    ## ratio, pm, pa true --M2 & M4
    est_obj2 = evaluator(sim_iid_dataset, q1_model1, q2_model1, q3_model1, q4_model1, q5_model1, ratio_model0, pm_model0,
                           reward_model1, pa_model0, eta_model1, target_policy = target_policy, control_policy = control_policy)
    
    est_obj2.estimate_DE_ME_SE()
    est_value2 = est_obj2.est_DEMESE
    var_value2 = est_obj2.var_DEMESE
    IDE_error_correct2 = est_value2[0] - true_IDE
    IME_error_correct2 = est_value2[1] - true_IME
    DDE_error_correct2 = est_value2[2] - true_DDE
    DME_error_correct2 = est_value2[3] - true_DME
    IDE_var_correct2 = var_value2[0] 
    IME_var_correct2 = var_value2[1] 
    DDE_var_correct2 = var_value2[2]
    DME_var_correct2 = var_value2[3]
    
    ## Q, eta, pm, reward -- M3
    est_obj3 = evaluator(sim_iid_dataset, q1_model0, q2_model0, q3_model0, q4_model0, q5_model0, ratio_model1, pm_model0,
                           reward_model0, pa_model1, eta_model0, target_policy = target_policy, control_policy = control_policy)
     
        
    est_obj3.estimate_DE_ME_SE()
    est_value3 = est_obj3.est_DEMESE
    var_value3 = est_obj3.var_DEMESE
    IDE_error_correct3 = est_value3[0] - true_IDE
    IME_error_correct3 = est_value3[1] - true_IME
    DDE_error_correct3 = est_value3[2] - true_DDE
    DME_error_correct3 = est_value3[3] - true_DME
    IDE_var_correct3 = var_value3[0] 
    IME_var_correct3 = var_value3[1] 
    DDE_var_correct3 = var_value3[2]
    DME_var_correct3 = var_value3[3]
    
    ## all false 
    est_obj4 = evaluator(sim_iid_dataset, q1_model1, q2_model1, q3_model1, q4_model1, q5_model1, ratio_model1, pm_model1,
                           reward_model1, pa_model1, eta_model1, target_policy = target_policy, control_policy = control_policy)
     
        
    est_obj4.estimate_DE_ME_SE()
    est_value4 = est_obj4.est_DEMESE
    var_value4 = est_obj4.var_DEMESE
    IDE_error_correct4 = est_value4[0] - true_IDE
    IME_error_correct4 = est_value4[1] - true_IME
    DDE_error_correct4 = est_value4[2] - true_DDE
    DME_error_correct4 = est_value4[3] - true_DME
    IDE_var_correct4 = var_value4[0] 
    IME_var_correct4 = var_value4[1] 
    DDE_var_correct4 = var_value4[2]
    DME_var_correct4 = var_value4[3]
    
    
    return [num_trajectory, num_time, seed,
            IDE_error_correct0,IME_error_correct0,DDE_error_correct0,DME_error_correct0,
            IDE_error_correct1,IME_error_correct1,DDE_error_correct1,DME_error_correct1,
            IDE_error_correct2,IME_error_correct2,DDE_error_correct2,DME_error_correct2,
            IDE_error_correct3,IME_error_correct3,DDE_error_correct3,DME_error_correct3,
            IDE_error_correct4,IME_error_correct4,DDE_error_correct4,DME_error_correct4,
            IDE_var_correct0,IME_var_correct0,DDE_var_correct0,DME_var_correct0, 
            IDE_var_correct1,IME_var_correct1,DDE_var_correct1,DME_var_correct1, 
            IDE_var_correct2,IME_var_correct2,DDE_var_correct2,DME_var_correct2, 
            IDE_var_correct3,IME_var_correct3,DDE_var_correct3,DME_var_correct3,
            IDE_var_correct4,IME_var_correct4,DDE_var_correct4,DME_var_correct4]

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
    
    with open("robust_final_test.txt","wb") as fp:
        pickle.dump(Error, fp)
