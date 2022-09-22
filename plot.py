from matplotlib.transforms import BlendedGenericTransform
import matplotlib.pyplot as plt
import seaborn as sns

def summary(out_df, N_range, T_range, absolute = True):
    result = []
    import pandas as pd
    import numpy as np
    for T in T_range:
        for N in N_range:
            a = np.vstack([seed_i[:15] for seed_i in out_df[T][N]])
            result.append(np.vstack([a[:,[12,13,14,i, i+1, i+2]] for i in [0,3]])) #'num_trajectory', 'num_time', 'seed', DE/ME/SE_error_TR/naive
    result = pd.DataFrame(np.vstack(result), columns = ["N","T","seed","DE_error","SE_error","ME_error"])
    result['NT'] = np.array(result)[:,0]*np.array(result)[:,1]
    result['DE_MSE'] = np.log(result['DE_error']**2)
    result['SE_MSE'] = np.log(result['SE_error']**2)
    result['ME_MSE'] = np.log(result['ME_error']**2)
    if absolute:
        result['DE_error'] = abs(result['DE_error'])
        result['SE_error'] = abs(result['SE_error'])
        result['ME_error'] = abs(result['ME_error'])
    NT_pairs = len(result.groupby(['N','T']).size())
    rep = int(len(result)/(2*NT_pairs))
    result["estimand"] = (["TR"]*rep+["naive"]*rep)*NT_pairs #[""]none*rep
    return result

def plot(result):
    fig, ((ax1, ax4) ,(ax2, ax5), (ax3, ax6)) = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=False, figsize = (10,5))

    COLORS = sns.color_palette("Set2")
    palette = {'TR' : COLORS[0],'naive' : COLORS[0]}
    style = {'TR' : (2,2),'naive' : (1,0)}

    ax1 = sns.lineplot(data=result
                         , x="NT", y="DE_error"
                         , hue="estimand" # group variable
                       , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax1
                        , n_boot = 20
                        , palette = palette
                       ,dashes=style
                       ,linewidth = 2.0
                        )
    ax1.set_title('abs(bias)')
    ax1.axes.set_ylabel("DE")
    
    ax2 = sns.lineplot(data=result
                         , x="NT", y="SE_error"
                         , hue="estimand" # group variable
                       , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax2
                        , n_boot = 20
                        , palette = palette
                       ,dashes=style
                       ,linewidth = 2.0
                        )
    #ax2.set_title('SE')
    ax2.axes.set_ylabel("SE")
    ax3 = sns.lineplot(data=result
                         , x="NT", y="ME_error"
                         , hue="estimand" # group variable
                       , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax3
                        , n_boot = 20
                        , palette = palette
                       ,dashes=style
                       ,linewidth = 2.0
                        )
    #ax3.set_title('ME')
    ax3.axes.set_ylabel("ME")


    ax4 = sns.lineplot(data=result
                         , x="NT", y="DE_MSE"
                         , hue="estimand" # group variable
                        , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax4
                        , n_boot = 20
                        , palette = palette
                       ,dashes=style
                       ,linewidth = 2.0
                        )
    ax4.set_title('logMSE')
    #ax4.set(yscale='log')
    #ax4.axes.set_ylabel()
    ax4.set(ylabel=None)
    
    ax5 = sns.lineplot(data=result
                         , x="NT", y="SE_MSE"
                         , hue="estimand" # group variable
                        , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax5
                        , n_boot = 20
                        , palette = palette
                       ,dashes=style
                       ,linewidth = 2.0
                        )
    ax5.set(ylabel=None)
    #ax5.set_title('SE')
    #ax5.set(yscale='log')
    #ax5.axes.set_ylabel()
    
    ax6 = sns.lineplot(data=result
                         , x="NT", y="ME_MSE"
                         , hue="estimand" # group variable
                        , style="estimand" 
                        , ci = 95
                        , err_style="bars"
                        , ax = ax6
                        , n_boot = 20
                        , palette = palette
                       ,dashes=style
                       ,linewidth = 2.0
                        )
    #ax6.set_title('ME')
    #ax6.set(yscale='log')
    #ax6.axes.set_ylabel()
    ax6.set(ylabel=None)

    title = fig.suptitle("Robustness of Effect Estimators", fontsize=15, y = 1.03)

    handles, labels = ax1.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='lower center', ncol = len(labels)
                           , bbox_to_anchor=(0.5, -0.1), bbox_transform=fig.transFigure)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()
    ax4.get_legend().remove()
    ax5.get_legend().remove()
    ax6.get_legend().remove()
    plt.tight_layout()
    plt.show()