from matplotlib.transforms import BlendedGenericTransform
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def summary(out_df, N_range, T_range, absolute = True):
    result = []
    for T in T_range:
        for N in N_range:
            a = np.vstack([seed_i[:12] for seed_i in out_df[T][N]])
            result.append(np.vstack([a[:,[9,10,11,i, i+1, i+2]] for i in [0,3,6]])) 
    result = pd.DataFrame(np.vstack(result), columns = ["N","T","seed","DE_error","ME_error","SE_error"])
    result['NT'] = np.array(result)[:,0]*np.array(result)[:,1]
    result['DE_MSE'] = np.log(result['DE_error']**2)
    result['ME_MSE'] = np.log(result['ME_error']**2)
    result['SE_MSE'] = np.log(result['SE_error']**2)
    NT_pairs = len(result.groupby(['N','T']).size())
    rep = int(len(result)/(3*NT_pairs))
    result["estimand"] = (["Triply-Robust"]*rep+["Direct"]*rep+["Baseline"]*rep)*NT_pairs #[""]none*rep
    
    if absolute:
        for N in N_range:
            for T in T_range:
                for error in [3,4,5]:
                    for estimand in ['Baseline', 'Direct', 'Triply-Robust']:
                        idx = result[(result['N'] == N) & (result['T'] == T) & (result['estimand']== estimand)].index
                        result.iloc[idx,error] = abs(result.iloc[idx,error].mean())
    return result
