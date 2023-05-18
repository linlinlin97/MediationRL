from matplotlib.transforms import BlendedGenericTransform
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def summary_IDEIME(out_df, N_range, T_range, absolute = True):
    result = []
    for T in T_range:
        for N in N_range:
            a = np.vstack([seed_i[:13] for seed_i in out_df[T][N]])
            result.append(np.vstack([a[:,[10,11,12,i, i+1]] for i in [0,2,4,6,8]])) 
    result = pd.DataFrame(np.vstack(result), columns = ["N","T","seed","IDE_error","IME_error"])
    result['NT'] = np.array(result)[:,0]*np.array(result)[:,1]
    result['IDE_MSE'] = np.log(result['IDE_error']**2)
    result['IME_MSE'] = np.log(result['IME_error']**2)
    NT_pairs = len(result.groupby(['N','T']).size())
    rep = int(len(result)/(5*NT_pairs))
    result["estimand"] = (["MR"]*rep+["DM"]*rep+["MIS1"]*rep+["MIS2"]*rep+["Baseline"]*rep)*NT_pairs #[""]none*rep
    
    if absolute:
        for N in N_range:
            for T in T_range:
                for error in [3,4]:
                    for estimand in ['Baseline', 'DM', 'MR', 'MIS1', 'MIS2']:
                        idx = result[(result['N'] == N) & (result['T'] == T) & (result['estimand']== estimand)].index
                        result.iloc[idx,error] = abs(result.iloc[idx,error].mean())
    return result

def summary(out_df, N_range, T_range, absolute = True, true = None):
    result = []
    for T in T_range:
        for N in N_range:
            a = np.vstack([seed_i[:23] for seed_i in out_df[T][N]])
            result.append(np.vstack([a[:,[-3,-2,-1,i, i+1, i+2, i+3]] for i in [0,4,8,12,16]])) #'num_trajectory', 'num_time', 'seed', DE/ME/SE_error_TR/naive/indep
    result = pd.DataFrame(np.vstack(result), columns = ["N","T","seed","IDE_error","IME_error","DDE_error","DME_error"])
    result['NT'] = np.array(result)[:,0]*np.array(result)[:,1]
    
    result[["IDE_error","IME_error","DDE_error","DME_error"]] -= true
    
    result['IDE_MSE'] = np.log(result['IDE_error']**2)
    result['IME_MSE'] = np.log(result['IME_error']**2)
    result['DDE_MSE'] = np.log(result['DDE_error']**2)
    result['DME_MSE'] = np.log(result['DME_error']**2)
    NT_pairs = len(result.groupby(['N','T']).size())
    rep = int(len(result)/(5*NT_pairs))
    result["estimand"] = (["MR"]*rep+["DM"]*rep+["MIS1"]*rep+["MIS2"]*rep+["Baseline"]*rep)*NT_pairs #[""]none*rep
    
    if absolute:
        for N in N_range:
            for T in T_range:
                for error in [3,4,5,6]:
                    for estimand in ['Baseline', 'DM', 'MR', 'MIS1', 'MIS2']:
                        idx = result[(result['N'] == N) & (result['T'] == T) & (result['estimand']== estimand)].index
                        result.iloc[idx,error] = np.log(abs(result.iloc[idx,error].mean()))
    return result


def plot(result, x='NT'):
    fig, ((ax1, ax2, ax3, ax4) ,(ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=False, figsize = (10,5))
    
    COLORS = sns.color_palette("colorblind")
    palette = {'MR' : COLORS[0],'DM' : COLORS[1], 'Baseline': COLORS[2], 'MIS1': COLORS[3], 'MIS2': COLORS[4]}

    ax1 = sns.lineplot(data=result, x=x, y="IDE_error", hue="estimand",
                       style="estimand", ci = 95, err_style="bars", ax = ax1,
                       n_boot = 20, palette = palette,linewidth = 2.0,
                       markers = True)
    ax1.set_title('IDE', fontsize = 18)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.axes.set_ylabel("logbias", fontsize = 18)
    
    ax2 = sns.lineplot(data=result, x=x, y="IME_error", hue="estimand",
                       style="estimand", ci = 95, err_style="bars", ax = ax2,
                       n_boot = 20, palette = palette,linewidth = 2.0,
                       markers = True)
    ax2.set_title('IME', fontsize = 18)
    ax2.tick_params(axis='y', labelsize=11)
    ax2.set(ylabel=None)
    
    ax3 = sns.lineplot(data=result, x=x, y="DDE_error", hue="estimand",
                       style="estimand", ci = 95, err_style="bars", ax = ax3,
                       n_boot = 20, palette = palette,linewidth = 2.0,
                       markers = True)
    ax3.set_title('DDE', fontsize = 18)
    ax3.tick_params(axis='y', labelsize=11)
    ax3.set(ylabel=None)
    
    ax4 = sns.lineplot(data=result, x=x, y="DME_error", hue="estimand",
                       style="estimand", ci = 95, err_style="bars", ax = ax4,
                       n_boot = 20, palette = palette,linewidth = 2.0,
                       markers = True)
    ax4.set_title('DME', fontsize = 18)
    ax4.tick_params(axis='y', labelsize=11)
    ax4.set(ylabel=None)

    ax5 = sns.lineplot(data=result, x=x, y="IDE_MSE", hue="estimand",
                       style="estimand", ci = 95, err_style="bars",
                       ax = ax5, n_boot = 20, palette = palette,
                       linewidth = 2.0,markers = True)
    ax5.axes.set_ylabel("logMSE", fontsize = 18)
    ax5.tick_params(axis='x', labelsize=11)
    ax5.axes.set_xlabel(x,fontsize = 12)
    
    ax6= sns.lineplot(data=result, x=x, y="IME_MSE", hue="estimand",
                       style="estimand", ci = 95, err_style="bars",
                       ax = ax6, n_boot = 20, palette = palette,
                       linewidth = 2.0,markers = True)
    ax6.set(ylabel=None)
    ax6.tick_params(axis='x', labelsize=11)
    ax6.axes.set_xlabel(x,fontsize = 12)
    
    ax7= sns.lineplot(data=result, x=x, y="DDE_MSE", hue="estimand",
                       style="estimand", ci = 95, err_style="bars",
                       ax = ax7, n_boot = 20, palette = palette,
                       linewidth = 2.0,markers = True)
    ax7.set(ylabel=None)
    ax7.tick_params(axis='x', labelsize=11)
    ax7.axes.set_xlabel(x,fontsize = 12)
    
    ax8= sns.lineplot(data=result, x=x, y="DME_MSE", hue="estimand",
                       style="estimand", ci = 95, err_style="bars",
                       ax = ax8, n_boot = 20, palette = palette,
                       linewidth = 2.0,markers = True)
    ax8.set(ylabel=None)
    ax8.tick_params(axis='x', labelsize=11)
    ax8.axes.set_xlabel(x,fontsize = 12)
    
    handles, labels = ax1.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='lower center', ncol = len(labels)
                           , bbox_to_anchor=(0.5, -0.1), bbox_transform=fig.transFigure,fontsize = 15)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()
    ax4.get_legend().remove()
    ax5.get_legend().remove()
    ax6.get_legend().remove()
    ax7.get_legend().remove()
    ax8.get_legend().remove()
    plt.tight_layout()
    plt.show()
    fig.savefig('Synthetic_Final_'+str(x), bbox_inches='tight')
    
    
def summary_baseline(out_df, N_range, T_range, absolute = True, true = None):
    result = []
    for T in T_range:
        for N in N_range:
            a = np.vstack([seed_i[:23] for seed_i in out_df[T][N]])
            result.append(np.vstack([a[:,[-3,-2,-1,i, i+1]] for i in [0,2,4]])) #'num_trajectory', 'num_time', 'seed', DE/ME/SE_error_TR/naive/indep
    result = pd.DataFrame(np.vstack(result), columns = ["N","T","seed","IDE_error","IME_error"])
    result['NT'] = np.array(result)[:,0]*np.array(result)[:,1]
    
    result[["IDE_error","IME_error"]] -= true
    
    result['IDE_MSE'] = np.log(result['IDE_error']**2)
    result['IME_MSE'] = np.log(result['IME_error']**2)
    NT_pairs = len(result.groupby(['N','T']).size())
    rep = int(len(result)/(3*NT_pairs))
    result["estimand"] = (["Baseline_DM"]*rep+["Baseline_IPW"]*rep+["Baseline_MR"]*rep)*NT_pairs #[""]none*rep
    
    if absolute:
        for N in N_range:
            for T in T_range:
                for error in [3,4]:
                    for estimand in ['Baseline_DM', 'Baseline_IPW', 'Baseline_MR']:
                        idx = result[(result['N'] == N) & (result['T'] == T) & (result['estimand']== estimand)].index
                        result.iloc[idx,error] = np.log(abs(result.iloc[idx,error].mean()))
    return result


def plot_baseline(result, x='NT'):
    fig, ((ax1, ax2, ax3, ax4) ,(ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=False, figsize = (10,5))
    
    COLORS = sns.color_palette("colorblind")
    palette = {'MR' : COLORS[0],'DM' : COLORS[1], 'MIS1': COLORS[3], 'MIS2': COLORS[4],'Baseline_IPW':COLORS[5],'Baseline_MR':COLORS[6],'Baseline_DM':COLORS[2]}

    ax1 = sns.lineplot(data=result, x=x, y="IDE_error", hue="estimand",
                       style="estimand", ci = 95, err_style="bars", ax = ax1,
                       n_boot = 20, palette = palette,linewidth = 2.0,
                       markers = True)
    ax1.set_title('IDE', fontsize = 18)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.axes.set_ylabel("bias", fontsize = 18)
    
    ax2 = sns.lineplot(data=result, x=x, y="IME_error", hue="estimand",
                       style="estimand", ci = 95, err_style="bars", ax = ax2,
                       n_boot = 20, palette = palette,linewidth = 2.0,
                       markers = True)
    ax2.set_title('IME', fontsize = 18)
    ax2.tick_params(axis='y', labelsize=11)
    ax2.set(ylabel=None)
    
    ax3 = sns.lineplot(data=result, x=x, y="DDE_error", hue="estimand",
                       style="estimand", ci = 95, err_style="bars", ax = ax3,
                       n_boot = 20, palette = palette,linewidth = 2.0,
                       markers = True)
    ax3.set_title('DDE', fontsize = 18)
    ax3.tick_params(axis='y', labelsize=11)
    ax3.set(ylabel=None)
    
    ax4 = sns.lineplot(data=result, x=x, y="DME_error", hue="estimand",
                       style="estimand", ci = 95, err_style="bars", ax = ax4,
                       n_boot = 20, palette = palette,linewidth = 2.0,
                       markers = True)
    ax4.set_title('DME', fontsize = 18)
    ax4.tick_params(axis='y', labelsize=11)
    ax4.set(ylabel=None)

    ax5 = sns.lineplot(data=result, x=x, y="IDE_MSE", hue="estimand",
                       style="estimand", ci = 95, err_style="bars",
                       ax = ax5, n_boot = 20, palette = palette,
                       linewidth = 2.0,markers = True)
    ax5.axes.set_ylabel("logMSE", fontsize = 18)
    ax5.tick_params(axis='x', labelsize=11)
    ax5.axes.set_xlabel(x,fontsize = 12)
    
    ax6= sns.lineplot(data=result, x=x, y="IME_MSE", hue="estimand",
                       style="estimand", ci = 95, err_style="bars",
                       ax = ax6, n_boot = 20, palette = palette,
                       linewidth = 2.0,markers = True)
    ax6.set(ylabel=None)
    ax6.tick_params(axis='x', labelsize=11)
    ax6.axes.set_xlabel(x,fontsize = 12)
    
    ax7= sns.lineplot(data=result, x=x, y="DDE_MSE", hue="estimand",
                       style="estimand", ci = 95, err_style="bars",
                       ax = ax7, n_boot = 20, palette = palette,
                       linewidth = 2.0,markers = True)
    ax7.set(ylabel=None)
    ax7.tick_params(axis='x', labelsize=11)
    ax7.axes.set_xlabel(x,fontsize = 12)
    
    ax8= sns.lineplot(data=result, x=x, y="DME_MSE", hue="estimand",
                       style="estimand", ci = 95, err_style="bars",
                       ax = ax8, n_boot = 20, palette = palette,
                       linewidth = 2.0,markers = True)
    ax8.set(ylabel=None)
    ax8.tick_params(axis='x', labelsize=11)
    ax8.axes.set_xlabel(x,fontsize = 12)
    
    handles, labels = ax1.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='lower center', ncol = len(labels)
                           , bbox_to_anchor=(0.5, -0.1), bbox_transform=fig.transFigure,fontsize = 15)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()
    ax4.get_legend().remove()
    ax5.get_legend().remove()
    ax6.get_legend().remove()
    ax7.get_legend().remove()
    ax8.get_legend().remove()
    plt.tight_layout()
    plt.show()
    fig.savefig('Synthetic_Final_Baselines_'+str(x), bbox_inches='tight')
    
    
def plot_toy2(result_iid, result_w_state, x='NT'):
    fig, ((ax1, ax2, ax3, ax4) ,(ax5, ax6,ax7,ax8)) = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=False, figsize = (10,5))

    COLORS = sns.color_palette("colorblind")
    palette = {'MR' : COLORS[0],'DM' : COLORS[1], 'MIS1': COLORS[3], 'MIS2': COLORS[4],'Baseline_IPW':COLORS[5],'Baseline_MR':COLORS[6],'Baseline_DM':COLORS[2]}

    ax1 = sns.lineplot(data=result_iid
                         , x=x, y="IDE_error"
                         , hue="estimand" # group variable
                       , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax1
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0,
                       markers = True
                        )
    ax1.set_title('IDE (i.i.d. S)', fontsize = 18)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.axes.set_ylabel("bias", fontsize = 18)
    
    ax2 = sns.lineplot(data=result_iid
                         , x=x, y="IME_error"
                         , hue="estimand" # group variable
                       , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax2
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0,
                       markers = True
                        )
    ax2.set_title('IME (i.i.d. S)', fontsize = 18)
    ax2.tick_params(axis='y', labelsize=11)
    ax2.set(ylabel=None)
    
    ax3 = sns.lineplot(data=result_w_state
                         , x=x, y="IDE_error"
                         , hue="estimand" # group variable
                       , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax3
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0,
                       markers = True
                        )
    ax3.set_title('IDE (w. S transit.)', fontsize = 18)
    ax3.tick_params(axis='y', labelsize=11)
    ax3.set(ylabel=None)
    
    ax4 = sns.lineplot(data=result_w_state
                         , x=x, y="IME_error"
                         , hue="estimand" # group variable
                       , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax4
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0,
                       markers = True
                        )
    ax4.set_title('IME (w. S transit.)', fontsize = 18)
    ax4.tick_params(axis='y', labelsize=11)
    ax4.set(ylabel=None)
    
    ax5 = sns.lineplot(data=result_iid
                         , x=x, y="IDE_MSE"
                         , hue="estimand" # group variable
                       , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax5
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0,
                       markers = True
                        )
    ax5.axes.set_ylabel("logMSE", fontsize = 18)
    ax5.tick_params(axis='x', labelsize=11)
    ax5.axes.set_xlabel(x,fontsize = 12)


    ax6 = sns.lineplot(data=result_iid
                         , x=x, y="IME_MSE"
                         , hue="estimand" # group variable
                        , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax6
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0,
                       markers = True
                        )
    ax6.set(ylabel=None)
    ax6.tick_params(axis='x', labelsize=11)
    ax6.axes.set_xlabel(x,fontsize = 12)
    
    
    ax7 = sns.lineplot(data=result_w_state
                         , x=x, y="IDE_MSE"
                         , hue="estimand" # group variable
                       , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax7
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0,
                       markers = True
                        )
    ax7.set(ylabel=None)
    ax7.tick_params(axis='x', labelsize=11)
    ax7.axes.set_xlabel(x,fontsize = 12)


    ax8 = sns.lineplot(data=result_w_state
                         , x=x, y="IME_MSE"
                         , hue="estimand" # group variable
                        , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax8
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0,
                       markers = True
                        )
    ax8.set(ylabel=None)
    ax8.tick_params(axis='x', labelsize=11)
    ax8.axes.set_xlabel(x,fontsize = 12)

    handles, labels = ax1.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='lower center', ncol = len(labels)
                           , bbox_to_anchor=(0.5, -0.1), bbox_transform=fig.transFigure,fontsize = 15)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()
    ax4.get_legend().remove()
    ax5.get_legend().remove()
    ax6.get_legend().remove()
    ax7.get_legend().remove()
    ax8.get_legend().remove()
    plt.tight_layout()
    plt.show()    
    fig.savefig('toy_2', bbox_inches='tight')