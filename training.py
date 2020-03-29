import os
import numpy as np
import pandas as pd
import argparse
from keras.models import load_model
from data import load_pfd, downsample
from model import training, HCCNN, VCCNN

### The file name for generate the new negative samples
# rolling
template_roll = ['FP20171019_0-1GHz_M31_dec-10arcmin_drifting_0483_DM45.30_1.00ms_Cand.pfd',
                 'FP20171027_0-1GHz_Dec-0701_drifting_0772_DM46.60_3.81ms_Cand.pfd',
                 'FP20171019_0-1GHz_M31_dec-10arcmin_drifting_0053_DM43.10_2.00ms_Cand.pfd',
                 'FP20171025_0-1GHz_Dec-0651_drifting_1002_DM38.70_1.00ms_Cand.pfd',
                 'FP20171029_0-1GHz_Dec-0641_drifting1_0719_DM40.40_2.00ms_Cand.pfd',
                 'FP20171030_0-1GHz_Dec-0721_drifting_0183_DM42.20_4.85ms_Cand.pfd',
                 'FP20171031_0-1GHz_Dec-0751_drifting_0450_DM13.30_2.00ms_Cand.pfd',
                 'FP20171031_0-1GHz_Dec-0751_drifting_0618_DM42.50_2.00ms_Cand.pfd',
                 'FP20171101_0-1GHz_Dec-0731_drifting_0036_DM21.30_2.05ms_Cand.pfd',
                 'FP20171107_0-1GHz_Dec-0831_drifting1_0422_DM146.00_2437.21ms_Cand.pfd',
                 'FP20171109_0-1GHz_Dec-0851_drifting_0035_DM43.80_2.00ms_Cand.pfd',
                 'FP20171118_0-1GHz_M31_SPH11-969_drifting_0325_DM29.10_1.84ms_Cand.pfd',
                 'FP20171118_0-1GHz_M31_SPH11-969_drifting_0733_DM34.30_2.00ms_Cand.pfd',
                 'FP20171120_0-1GHz_M31_SPH11-182_drifting_0618_DM13.50_4.00ms_Cand.pfd',
                 'FP20171123_0-1GHz_M31_SPH11-883_drifting_0655_DM320.00_3.80ms_Cand.pfd',
                 'FP20171129_0-1GHz_C17+10arcmin_drifting_0857_DM99.10_39.99ms_Cand.pfd',
                 'FP20171111_0-1GHz_Dec-0901_drifting_0451_DM31.30_2.00ms_Cand.pfd']

template_random = ['FP20171028_0-1GHz_Dec-0651_drifting_0614_DM46.20_2.15ms_Cand.pfd',
    'FP20171027_0-1GHz_Dec-0701_drifting_0524_DM39.70_3.74ms_Cand.pfd',
    'FP20171129_0-1GHz_C17+10arcmin_drifting_0857_DM99.10_39.99ms_Cand.pfd',
    'FP20180503_0-1GHz_Dec+40.6_drifting_0293_DM22.80_2.00ms_Cand.pfd',
    'FP20180429_0-1GHz_Dec+26.4_drifting_0041_DM34.10_2.00ms_Cand.pfd',
    'FP20180426_0-1GHz_Dec+39.7_drifting_0294_DM49.20_3.00ms_Cand.pfd',
    'FP20180426_0-1GHz_Dec+39.7_drifting_0052_DM10.60_2.08ms_Cand.pfd',
    'FP20180424_0-1GHz_Dec+39.9_drifting_0475_DM205.50_2.49ms_Cand.pfd',
    'FP20180423_0-1GHz_Dec+47.4_drifting_0703_DM100.60_2.00ms_Cand.pfd',
    'FP20180423_0-1GHz_Dec+47.4_drifting_0591_DM407.00_2.67ms_Cand.pfd',
    'FP20180423_0-1GHz_Dec+47.4_drifting_0413_DM336.00_2.67ms_Cand.pfd',
    'FP20171018_0-1GHz_4C-06.18_drifting_0077_DM10.50_3.45ms_Cand.pfd',
    'FP20171018_0-1GHz_4C-06.18_drifting_0749_DM33.70_1.00ms_Cand.pfd',
    'FP20171018_0-1GHz_4C-06.18_drifting_0751_DM18.80_1.05ms_Cand.pfd',
    'FP20171025_0-1GHz_Dec-0651_drifting_0526_DM307.00_1.87ms_Cand.pfd',
    'FP20171025_0-1GHz_Dec-0651_drifting_0602_DM10.40_0.89ms_Cand.pfd',
    'FP20171027_0-1GHz_Dec-0701_drifting_0290_DM91.30_1.92ms_Cand.pfd',
    'FP20171027_0-1GHz_Dec-0701_drifting_0524_DM39.70_3.74ms_Cand.pfd',
    'FP20171028_0-1GHz_Dec-0651_drifting_0614_DM46.20_2.15ms_Cand.pfd',
    'FP20171107_0-1GHz_Dec-0831_drifting1_0466_DM385.00_2.26ms_Cand.pfd',
    'FP20171107_0-1GHz_Dec-0831_drifting1_0712_DM39.00_2.00ms_Cand.pfd',
    'FP20171108_0-1GHz_Dec-0841_drifting_0034_DM868.00_2.67ms_Cand.pfd',
    'FP20171122_0-1GHz_M31_SPH11-1234_drifting_0200_DM766.00_1.84ms_Cand.pfd']

template_subints = ['FP20171102_0-1GHz_Dec-0741_drifting_0142_DM816.00_2278.26ms_Cand.pfd',
                    'FP20171128_0-1GHz_M31_SPH11-1066+10arcmin_drifting_0666_DM15.70_9.09ms_Cand.pfd',
                    'FP20180109_0-1GHz_Dec+43.15_drifting_0690_DM49.10_3.76ms_Cand.pfd',
                    'FP20180214_0-1GHz_Dec+41.1_drifting1_0195_DM196.50_17.71ms_Cand.pfd',
                    'FP20180302_0-1GHz_Dec+39.6_drifting_0296_DM34.00_3.79ms_Cand.pfd',
                    'FP20171108_0-1GHz_Dec-0841_drifting_0466_DM44.20_1.29ms_Cand.pfd',
                    'FP20171115_0-1GHz_Dec-0941_drifting1_0391_DM16.40_1.99ms_Cand.pfd',
                    'FP20171117_0-1GHz_M31_SPH11-1040_drifting_0283_DM45.30_2.75ms_Cand.pfd',
                    'FP20171120_0-1GHz_M31_SPH11-182_drifting_0502_DM44.60_2.05ms_Cand.pfd',
                    'FP20171123_0-1GHz_M31_SPH11-883_drifting_0237_DM44.70_2.13ms_Cand.pfd',
                    'FP20171128_0-1GHz_M31_SPH11-1066+10arcmin_drifting_0500_DM10.60_1.65ms_Cand.pfd',
                    'FP20171128_0-1GHz_M31_SPH11-1066+10arcmin_drifting_0568_DM20.50_2.85ms_Cand.pfd']


template_subints2 = ['FP20171018_0-1GHz_4C-06.18_drifting_0323_DM45.40_1.03ms_Cand.pfd',
                     'FP20171018_0-1GHz_4C-06.18_drifting_0497_DM44.90_2.00ms_Cand.pfd',
                     'FP20171018_0-1GHz_4C-06.18_drifting_0749_DM33.70_1.00ms_Cand.pfd',
                     'FP20171018_0-1GHz_4C-06.18_drifting_0751_DM18.80_1.05ms_Cand.pfd',
                     'FP20171019_0-1GHz_M31_dec-10arcmin_drifting_0075_DM25.00_1.04ms_Cand.pfd',
                     'FP20171019_0-1GHz_M31_dec-10arcmin_drifting_0269_DM43.10_2.00ms_Cand.pfd',
                     'FP20171019_0-1GHz_M31_dec-10arcmin_drifting_0291_DM22.90_1.49ms_Cand.pfd',
                     #'FP20171019_0-1GHz_M31_dec-10arcmin_drifting_0309_DM43.10_2.00ms_Cand.pfd',
                     #'FP20171019_0-1GHz_M31_dec-10arcmin_drifting_0365_DM37.50_0.96ms_Cand.pfd',
                     'FP20171019_0-1GHz_M31_dec-10arcmin_drifting_0483_DM45.30_1.00ms_Cand.pfd',
                     'FP20171024_0-1GHz_Dec-0701_drifting_0826_DM38.70_1.00ms_Cand.pfd',
                     'FP20171024_0-1GHz_Dec-0701_drifting_0950_DM204.00_2.00ms_Cand.pfd',
                     'FP20171027_0-1GHz_Dec-0701_drifting_0552_DM23.60_0.81ms_Cand.pfd',
                     'FP20171027_0-1GHz_Dec-0701_drifting_0606_DM218.50_9.96ms_Cand.pfd',
                     'FP20171027_0-1GHz_Dec-0701_drifting_0678_DM29.80_0.92ms_Cand.pfd',
                     'FP20171027_0-1GHz_Dec-0701_drifting_0748_DM21.90_1.18ms_Cand.pfd',
                     'FP20171029_0-1GHz_Dec-0641_drifting1_0719_DM40.40_2.00ms_Cand.pfd',
                     'FP20171030_0-1GHz_Dec-0721_drifting_0197_DM40.60_1.19ms_Cand.pfd',
                     'FP20171031_0-1GHz_Dec-0751_drifting_0450_DM13.30_2.00ms_Cand.pfd',
                     'FP20171031_0-1GHz_Dec-0751_drifting_0618_DM42.50_2.00ms_Cand.pfd',
                     'FP20171031_0-1GHz_Dec-0751_drifting_0654_DM40.30_1.13ms_Cand.pfd',
                     'FP20171101_0-1GHz_Dec-0731_drifting_0036_DM21.30_2.05ms_Cand.pfd',
                     'FP20171101_0-1GHz_Dec-0731_drifting_0076_DM38.80_1.48ms_Cand.pfd',
                     'FP20171101_0-1GHz_Dec-0731_drifting_0356_DM35.70_2.00ms_Cand.pfd',
                     'FP20171101_0-1GHz_Dec-0731_drifting_0418_DM31.10_1.48ms_Cand.pfd',
                     'FP20171101_0-1GHz_Dec-0731_drifting_0424_DM37.60_1.48ms_Cand.pfd',
                     'FP20171101_0-1GHz_Dec-0731_drifting_0514_DM13.30_1.00ms_Cand.pfd',
                     'FP20171101_0-1GHz_Dec-0731_drifting_0672_DM33.70_1.00ms_Cand.pfd',
#                     'FP20171102_0-1GHz_Dec-0741_drifting_0028_DM1129.00_5.03ms_Cand.pfd',
#                     'FP20171102_0-1GHz_Dec-0741_drifting_0090_DM2206.00_5.00ms_Cand.pfd',
                     'FP20171102_0-1GHz_Dec-0741_drifting_0112_DM204.00_10.06ms_Cand.pfd',
#                     'FP20171102_0-1GHz_Dec-0741_drifting_0146_DM1324.00_9.96ms_Cand.pfd',
#                     'FP20171103_0-1GHz_Dec-0801_drifting_0103_DM2181.00_9.94ms_Cand.pfd'
                     'FP20171103_0-1GHz_Dec-0801_drifting_0105_DM45.50_2.48ms_Cand.pfd',
                     'FP20171103_0-1GHz_Dec-0801_drifting_0255_DM144.50_2.51ms_Cand.pfd',
                     'FP20171103_0-1GHz_Dec-0801_drifting_0487_DM38.90_2.00ms_Cand.pfd',
                     'FP20171103_0-1GHz_Dec-0801_drifting_0533_DM42.00_2.00ms_Cand.pfd',
                     'FP20171105_0-1GHz_Dec-0821_drifting_0134_DM279.00_2278.26ms_Cand.pfd',
                     'FP20171105_0-1GHz_Dec-0821_drifting_0278_DM16.80_1.87ms_Cand.pfd',
                     'FP20171105_0-1GHz_Dec-0821_drifting_0514_DM46.40_2.00ms_Cand.pfd',
                     'FP20171105_0-1GHz_Dec-0821_drifting_0544_DM278.00_1.78ms_Cand.pfd',
                     'FP20171105_0-1GHz_Dec-0821_drifting_0554_DM21.10_1.00ms_Cand.pfd',
                     'FP20171105_0-1GHz_Dec-0821_drifting_0608_DM58.00_2.41ms_Cand.pfd',
                     #'FP20171107_0-1GHz_Dec-0831_drifting1_0466_DM385.00_2.26ms_Cand.pfd',
                     #'FP20171107_0-1GHz_Dec-0831_drifting1_0538_DM29.20_1.58ms_Cand.pfd',
                     #'FP20171107_0-1GHz_Dec-0831_drifting1_0562_DM17.20_1.00ms_Cand.pfd',
                     #'FP20171107_0-1GHz_Dec-0831_drifting1_0562_DM41.70_1.57ms_Cand.pfd'
                     ]


def rolling_subbands(ori_subbands):
    roll_index = np.random.randint(10, 54)
    subbands = np.roll(ori_subbands, roll_index, axis=0)
    # subbdnas = subbands + np.random.normal(0, 0.1, subbands.shape)
    # subbands = subbands - subbands.min()
    return subbands

def rfi_subbands(ori_subbands):
    '''
    To insert the narrow band rfi to the pulsar pfd
    '''
    N = np.arange(64) - 31
    def wd(n, sigma_n):
        return np.exp(-n**2 / (2*sigma_n**2)) / (sigma_n * np.sqrt(2*np.pi))
    
    num_rfi = np.random.randint(1, 4)
    decay_ratio = np.clip(np.random.normal(0.5, 0.1, 1), 0.1, 1)
    subbands = ori_subbands * decay_ratio
    for i in range(num_rfi):
        rfi_index = np.random.randint(10, 53)
        sigma = np.random.uniform(0.3, 0.8)
        rfi = wd(N*0.08, sigma) + np.random.normal(0, 0.15, 64)
        rfi = downsample(rfi[::2], 64)
    
        subbands[rfi_index] = subbands[rfi_index] + rfi
    
    subbands = np.clip(subbands, 0, None)
    
    return subbands




if __name__ == '__main__':
    # model
    model_mode = 'HCCNN'


    path = ['PICS-ResNet_data/train_data/pulsar', 'PICS-ResNet_data/train_data/rfi']
    fpath_list = []
    fname_list = []
    pfd_data = []
    num_list = []

    for path_item in path:
        path_item_temp = os.path.join(path_item, '*.pfd')
        fpath_list_temp = glob(path_item_temp)
        num_list.append(len(fpath_list_temp))
        fpath_list.extend(fpath_list_temp)
    else:
        fpath_list = glob.glob(path)
    
    for f in fpath_list:
        pfd_data.append(load_pfd(f))
        fname_list.append(f.split('/')[-1])

    pfd_data = pd.DataFrame(pfd_data, index=fname_list)
    y = np.concatenate([np.ones(num_list[0]), np.zeros(num_list[1])])
    # generate the false dm curve samples from training pulsar data

    num_gen_false_dm = 50
    num_train_pulsar = int(np.sum(y))
    real_index = np.random.permutation(num_train_pulsar)[:num_gen_false_dm]
    pfd_DM = pfd_data.iloc[real_index].copy()
    # error_fname = 'FP20180418_0-1GHz_Dec+21.55_drifting_0705_DM79.60_352.75ms_Cand.pfd'
    new_pfd_name = []
    for i in range(len(pfd_DM)):
        new_pfd_name.append(pfd_DM.index[0] + '.neg_DM')
        # update subbands(gaussian noise)
        subbands = pfd_DM.subbands.iloc[i].copy()
        #subbands = subbands + np.random.normal(0, 0.1, subbands.shape)
        #subbands -= subbands.min()
        pfd_DM.subbands.iloc[i] = subbands
        
        #update subints(gaussian noise)
        subints = pfd_DM.time_vs_phase.iloc[i].copy()
        #subints = subints + np.random.normal(0, 0.1, subints.shape)
        #subints -= subints.min()
        pfd_DM.time_vs_phase.iloc[i] = subints

        # update DM
        DM = cut_DM(pfd_DM.DM.iloc[i].copy())
        DM = DM - DM.min()
        if DM.max() != 0:
            DM = DM * 1.0 / np.max(DM)
        else:
            DM = DM
        pfd_DM.DM.iloc[i] = DM
        
    pfd_DM.index = new_pfd_name


    rest_index = set(list(np.arange(len(pfd_data)))) - set(real_index) - set()
    num_subbands_rolling = len(template_roll)
    num_subbands_random = len(template_random)
    num_gen_subbands = 2 * num_subbands_rolling + 2 * num_subbands_random
    temp_index = np.random.permutation(len(rest_index))[:num_gen_subbands]
    gen_subbands_index = np.array(list(rest_index))[temp_index]

    pfd_subbands = pfd_data.iloc[gen_subbands_index].copy()
    # error_fname = 'FP20180418_0-1GHz_Dec+21.55_drifting_0705_DM79.60_352.75ms_Cand.pfd'
    new_pfd_name = []
    for i in range(num_subbands_rolling):
        new_pfd_name.append(pfd_subbands.index[i] + '.neg_subbands')
        # update subbands(gaussian noise)
        template_fname = template_roll[i]
        subbands = pfd_data.loc[template_fname].subbands.copy()
        subbands = rolling_subbands(subbands)
        pfd_subbands.subbands.iloc[i] = subbands
        
        #update subints(gaussian noise)
        subints = pfd_subbands.time_vs_phase.iloc[i].copy()
        #subints = subints + np.random.normal(0, 0.1, subints.shape)
        #subints -= subints.min()
        pfd_subbands.time_vs_phase.iloc[i] = subints
        
    for i in range(num_subbands_rolling, 2*num_subbands_rolling):
        new_pfd_name.append(pfd_subbands.index[i] + '.neg_subbands')
        # update subbands(gaussian noise)
        template_fname = template_roll[i - num_subbands_rolling]
        subbands = pfd_data.loc[template_fname].subbands.copy()
        # subbands = rolling_subbands(subbands)
        pfd_subbands.subbands.iloc[i] = subbands

        #update subints(gaussian noise)
        subints = pfd_subbands.time_vs_phase.iloc[i].copy()
        #subints = subints + np.random.normal(0, 0.1, subints.shape)
        #subints -= subints.min()
        pfd_subbands.time_vs_phase.iloc[i] = subints    


    for i in range(2 * num_subbands_rolling, 2 * num_subbands_rolling + num_subbands_random):
        new_pfd_name.append(pfd_subbands.index[i] + '.neg_subbands')
        # update subbands(gaussian noise)
        template_fname = template_random[i - 2 * num_subbands_rolling]
        subbands = pfd_data.loc[template_fname].subbands.copy()
        #subbands = subbands + np.random.normal(0, 0.1, subbands.shape)
        #subbands -= subbands.min()
        pfd_subbands.subbands.iloc[i] = subbands
        
        #update subints(gaussian noise)
        subints = pfd_subbands.time_vs_phase.iloc[i].copy()
        #subints = subints + np.random.normal(0, 0.1, subints.shape)
        #subints -= subints.min()
        pfd_subbands.time_vs_phase.iloc[i] = subints
        
    for i in range(2 * num_subbands_rolling + num_subbands_random, num_gen_subbands):
        new_pfd_name.append(pfd_subbands.index[i] + '.neg_subbands')
        # update subbands(gaussian noise)
        template_fname = template_random[i - 2 * num_subbands_rolling - num_subbands_random]
        subbands = pfd_data.loc[template_fname].subbands.copy()
        subbands = rfi_subbands(subbands)
        pfd_subbands.subbands.iloc[i] = subbands
        
        
        #update subints(gaussian noise)
        subints = pfd_subbands.time_vs_phase.iloc[i].copy()
        #subints = subints + np.random.normal(0, 0.1, subints.shape)
        #subints -= subints.min()
        pfd_subbands.time_vs_phase.iloc[i] = subints
        

    pfd_subbands.index = new_pfd_name

    rest_index = rest_index - set(gen_subbands_index)
    num_template = len(template_subints)
    num_gen_subints = 2 * num_template
    temp_index = np.random.permutation(len(rest_index))[:num_gen_subints]
    gen_subints_index = np.array(list(rest_index))[temp_index]

    pfd_subints = pfd_data.iloc[gen_subints_index].copy()

    new_pfd_name = []

    for i in range(num_template):
        new_pfd_name.append(pfd_subints.index[i] + '.neg_subints')
        template_fname = template_subints[i]
        subints = pfd_data.loc[template_fname].time_vs_phase.copy()
        pfd_subints.time_vs_phase.iloc[i] = subints
        sumprof = subints.mean(axis=0)
        normprof = sumprof - sumprof.min()
        if normprof.max() == 0:
            sumprof = normprof
        else:
            sumprof = normprof / normprof.max()
        pfd_subints.sumprof.iloc[i] = sumprof

    for i in range(num_template, num_gen_subints):
        new_pfd_name.append(pfd_subints.index[i] + '.neg_subints')
        template_fname = template_subints[i-num_template]
        subints = pfd_data.loc[template_fname].time_vs_phase.copy()
        subints = subints + np.random.normal(0, 0.1, subints.shape)
        subints -= subints.min()
        pfd_subints.time_vs_phase.iloc[i] = subints
        sumprof = subints.mean(axis=0)
        normprof = sumprof - sumprof.min()
        if normprof.max() == 0:
            sumprof = normprof
        else:
            sumprof = normprof / normprof.max()
        pfd_subints.sumprof.iloc[i] = sumprof


    pfd_subints.index = new_pfd_name


    rest_index = rest_index - set(gen_subints_index)
    num_template = len(template_subints2)
    num_gen_subints2 = 2 * num_template
    temp_index = np.random.permutation(len(rest_index))[:num_gen_subints2]
    gen_subints_index2 = np.array(list(rest_index))[temp_index]

    pfd_subints2 = pfd_data.iloc[gen_subints_index2].copy()

    new_pfd_name = []

    for i in range(num_template):
        new_pfd_name.append(pfd_subints2.index[i] + '.neg_subints')
        template_fname = template_subints2[i]
        subints = pfd_data.loc[template_fname].time_vs_phase.copy()
        pfd_subints2.time_vs_phase.iloc[i] = subints
        sumprof = subints.mean(axis=0)
        normprof = sumprof - sumprof.min()
        if normprof.max() == 0:
            sumprof = normprof
        else:
            sumprof = normprof / normprof.max()
        pfd_subints2.sumprof.iloc[i] = sumprof
        subbands = pfd_data.loc[template_fname].subbands.copy()
        pfd_subints2.subbands.iloc[i] = subbands

    for i in range(num_template, num_gen_subints2):
        new_pfd_name.append(pfd_subints2.index[i] + '.neg_subints')
        template_fname = template_subints2[i-num_template]
        subints = pfd_data.loc[template_fname].time_vs_phase.copy()
        subints = subints + np.random.normal(0, 0.1, subints.shape)
        subints -= subints.min()
        pfd_subints2.time_vs_phase.iloc[i] = subints
        sumprof = subints.mean(axis=0)
        normprof = sumprof - sumprof.min()
        if normprof.max() == 0:
            sumprof = normprof
        else:
            sumprof = normprof / normprof.max()
        pfd_subints2.sumprof.iloc[i] = sumprof
        subbands = pfd_data.loc[template_fname].subbands.copy()
        subbands = subbands + np.random.normal(0, 0.1, subbands.shape)
        subbands -= subbands.min()
        pfd_subints2.subbands.iloc[i] = subbands


    pfd_subints2.index = new_pfd_name

    pfd_data = pd.concat([pfd_data, pfd_DM, pfd_subbands, pfd_subints, pfd_subints2])
    y = np.concatenate([y, np.zeros(num_gen_false_dm), np.zeros(num_gen_subbands), 
        np.zeros(num_gen_subints), np.zeros(num_gen_subints2)])

    if model_mode == 'HCCNN':
        model = HCCNN()
    else:
        model = VCCNN()

    # training the model
    model = traning(model, pfd_data, y, model_mode=model_mode)
    
    model.save(model_mode+'.h5')
