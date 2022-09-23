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
            result.append(np.vstack([a[:,[9,10,11,i, i+1, i+2]] for i in [0,3,6]])) #'num_trajectory', 'num_time', 'seed', DE/ME/SE_error_TR/naive/indep
    result = pd.DataFrame(np.vstack(result), columns = ["N","T","seed","DE_error","ME_error","SE_error"])
    result['NT'] = np.array(result)[:,0]*np.array(result)[:,1]
    result['DE_MSE'] = result['DE_error']**2
    result['ME_MSE'] = result['ME_error']**2
    result['SE_MSE'] = result['SE_error']**2
    if absolute:
        result['DE_error'] = abs(result['DE_error'])
        result['ME_error'] = abs(result['ME_error'])
        result['SE_error'] = abs(result['SE_error'])
    NT_pairs = len(result.groupby(['N','T']).size())
    rep = int(len(result)/(3*NT_pairs))
    result["estimand"] = (["Triply-Robust"]*rep+["Direct"]*rep+["Baseline"]*rep)*NT_pairs #[""]none*rep
    #else: other test ndim/MCMC
    #    rep = int(len(result)/len(MCMC_range)/2)
    #    result["estimand"] = (["TR"]*rep+["naive"]*rep)*len(MCMC_range)
    #    result['MCMC'] = np.hstack([[i]*rep*2 for i in MCMC_range])
    return result

def plot(result, x='NT'):
    fig, ((ax1, ax4) ,(ax2, ax5), (ax3, ax6)) = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=False, figsize = (10,5))

    COLORS = sns.color_palette("Set2")
    palette = {'Triply-Robust' : COLORS[0],'Direct' : COLORS[1], 'Baseline': COLORS[2]}
    style = {'Triply-Robust' : (2,2),'Direct' : (1,0), 'Baseline': (.5,.5)}

    ax1 = sns.lineplot(data=result
                         , x=x, y="DE_error"
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
                         , x=x, y="ME_error"
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
    #ax3.set_title('ME')
    ax2.axes.set_ylabel("ME")
    
    ax3 = sns.lineplot(data=result
                         , x=x, y="SE_error"
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
    #ax2.set_title('SE')
    ax3.axes.set_ylabel("SE")


    ax4 = sns.lineplot(data=result
                         , x=x, y="DE_MSE"
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
                         , x=x, y="ME_MSE"
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
    #ax6.set_title('ME')
    #ax6.set(yscale='log')
    #ax6.axes.set_ylabel()
    ax5.set(ylabel=None)
    
    ax6 = sns.lineplot(data=result
                         , x=x, y="SE_MSE"
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
    ax6.set(ylabel=None)
    #ax5.set_title('SE')
    #ax5.set(yscale='log')
    #ax5.axes.set_ylabel()

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