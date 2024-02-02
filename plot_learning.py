import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

font_size = 6
label_size = 9

n_fig_per_row = 2
textwidth_pt = 487.8225
points_per_inch = 72.27
textwidth_inches = textwidth_pt / points_per_inch
image_size = textwidth_inches / n_fig_per_row
golden = (1 + 5 ** 0.5) / 2

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",  # Choose the same font family as in your LaTeX document
# })
rcParams.update({'figure.autolayout': True})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

M = 10
idx = 0
N_set = [308,768]
D_set = [6,8]
for name in ["yacht","energy"]:
    data = np.load(name+".npz")
    P_CPD = data['P_CPD']
    mean_RMSE_CPD = data['mean_RMSE_CPD']
    std_RMSE_CPD = data['std_RMSE_CPD']

    P_TT = data['P_TT']
    mean_RMSE_TT = data['mean_RMSE_TT']
    std_RMSE_TT = data['std_RMSE_TT']

    N = N_set[idx]
    D = D_set[idx]

    plt.figure(figsize=(image_size,image_size/golden))
    plt.rc('font', size=font_size)

    plt.grid(True, linestyle='dashed', axis='y', linewidth=0.1, which = 'both', dashes=(10, 5))
    plt.plot(P_CPD,mean_RMSE_CPD,label='CPD',linewidth=1,zorder=2)
    plt.plot(P_TT,mean_RMSE_TT,label='TT',linewidth=1,zorder=4)
    current_xlim = plt.xlim()
    plt.hlines(data['mean_RMSE_GP'],-100,2*max(np.maximum(P_CPD,P_TT)),label='GP',colors='black',zorder=0,linewidth=1)
    plt.fill_between(P_CPD,mean_RMSE_CPD-std_RMSE_CPD,mean_RMSE_CPD+std_RMSE_CPD,alpha=0.5,zorder=1)
    plt.fill_between(P_TT,mean_RMSE_TT-std_RMSE_TT,mean_RMSE_TT+std_RMSE_TT,alpha=0.5,zorder=3)
    plt.xlim(current_xlim)
    if idx == 0:
        plt.ylabel("RMSE",fontsize=label_size,loc='center')
    if idx == 1:
        plt.legend(fontsize=font_size)
    plt.xlabel(r'$P$',fontsize=label_size,loc='center')
    plt.title(f'{name}\n$N={N}$, $D={D}$', fontsize=label_size)
    # plt.savefig(name+".pdf",bbox_inches="tight",pad_inches=0)
    plt.close()
    idx += 1