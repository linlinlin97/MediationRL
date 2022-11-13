from matplotlib.transforms import BlendedGenericTransform
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def summary(out_df, N_range, T_range):
    result = []
    for T in T_range:
        for N in N_range:
            a = np.vstack([seed_i for seed_i in out_df[T][N]])
            result.append(np.vstack([a[:,[0,1,2,i, i+1, i+2]] for i in [3,6,9,12,15]])) #'num_trajectory', 'num_time', 'seed', DE/ME/SE_error_TR/naive/indep
    result = pd.DataFrame(np.vstack(result), columns = ["N","T","seed","DE_error","ME_error","SE_error"])
    result['NT'] = np.array(result)[:,0]*np.array(result)[:,1]
    result['DE_MSE'] = np.log(result['DE_error']**2)
    result['ME_MSE'] = np.log(result['ME_error']**2)
    result['SE_MSE'] = np.log(result['SE_error']**2)
    NT_pairs = len(result.groupby(['N','T']).size())
    rep = int(len(result)/(5*NT_pairs))
    result["correct_model"] = (["M1& M2 & M3"]*rep+["M1"]*rep+["M2"]*rep+["M3"]*rep+["None"]*rep)*NT_pairs#[""]none*rep
    
    return result

def plot(result, x='NT'):
    fig, ((ax1, ax2, ax3) ,(ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False, figsize = (10,5))
    
    COLORS = sns.color_palette("colorblind")
    palette = {"M1& M2 & M3":COLORS[0],"M1":COLORS[5],"M2":COLORS[2],"M3":COLORS[3],"None":COLORS[4]}
    #COLORS = sns.color_palette("Set2")
    #palette = {name : color for name, color in zip(["M1& M2 & M3","M1","M2","M3","None"], COLORS)}

    ax1 = sns.lineplot(data=result
                         , x=x, y="DE_error"
                         , hue="correct_model" # group variable
                       , style="correct_model"
                        , ci = 95
                        , err_style="bars"
                        , ax = ax1
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0
                       ,markers = True
                        )
    ax1.set_title('DE')
    ax1.axes.set_ylabel("bias")
    
    ax2 = sns.lineplot(data=result
                         , x=x, y="ME_error"
                         , hue="correct_model" # group variable
                       , style="correct_model"
                        , ci = 95
                        , err_style="bars"
                        , ax = ax2
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0
                       ,markers = True
                        )
    ax2.set_title('ME')
    #ax2.axes.set_ylabel("ME")
    ax2.set(ylabel=None)
    
    ax3 = sns.lineplot(data=result
                         , x=x, y="SE_error"
                         , hue="correct_model"# group variable
                       , style="correct_model"
                        , ci = 95
                        , err_style="bars"
                        , ax = ax3
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0
                       ,markers = True
                        )
    ax3.set_title('SE')
    #ax3.axes.set_ylabel("SE")
    ax3.set(ylabel=None)


    ax4 = sns.lineplot(data=result
                         , x=x, y="DE_MSE"
                         , hue="correct_model" # group variable
                        , style="correct_model" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax4
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0
                       ,markers = True
                        )
    ax4.axes.set_ylabel("logMSE")
    
    
    ax5 = sns.lineplot(data=result
                         , x=x, y="ME_MSE"
                         , hue="correct_model"# group variable
                        , style="correct_model"
                        , ci = 95
                        , err_style="bars"
                        , ax = ax5
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0
                       ,markers = True
                        )

    ax5.set(ylabel=None)
    
    ax6 = sns.lineplot(data=result
                         , x=x, y="SE_MSE"
                         , hue="correct_model" # group variable
                        , style="correct_model"
                        , ci = 95
                        , err_style="bars"
                        , ax = ax6
                        , n_boot = 20
                        , palette = palette
                       ,linewidth = 2.0
                       ,markers = True
                        )
    ax6.set(ylabel=None)


    handles, labels = ax1.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='lower center', ncol = len(labels)
                           , bbox_to_anchor=(0.5, -0.1), bbox_transform=fig.transFigure,fontsize = 15)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()
    ax4.get_legend().remove()
    ax5.get_legend().remove()
    ax6.get_legend().remove()
    plt.tight_layout()
    plt.show()
    fig.savefig('robust_Final', bbox_inches='tight')