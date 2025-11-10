##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0
##########################################################################################
# Path Config
import os
import sys
import json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
##########################################################################################
# import
import logging
import numpy as np
from LEHD_TSP_p2.utils.utils import create_logger
from LEHD_TSP_p2.TSP.TSPTester_inTSPlib import TSPTester as Tester

########### Frequent use parameters  ##################################################

test_in_tsplib = True  # test in tsplib or not
Use_RRC = False         # decode method: use RRC or not (greedy)
RRC_budget = 50         # RRC budget

########### model to load ###############

model_load_path = './LEHD_TSP_p2/TSP/result/saved_p2_checkpoints'   # saved models with progressive finetuning 
model_load_epoch = 3

##########################################################################################
mode = 'test'
test_paras = {
   # problem_size: [filename, episode, batch]
    0: ['TSPlib_70instances.txt', 70, 1],
    'R': ['tsplib_typeR.txt',19,1],
    'RC': ['tsplib_typeRC.txt',12,1],
    'C': ['tsplib_typeC.txt', 17, 1],
    'mixed':['tsplib_train.txt', 48, 1]
}

if not Use_RRC:
    RRC_budget = 0

##########################################################################################

b = os.path.abspath("./LEHD_TSP_p2/TSP") 
env_params = {
    'mode': mode,
    'test_in_tsplib':test_in_tsplib,
    'tsplib_path':  b + f"/data/{test_paras[0][0]}",
    'data_path':  b + f"/data/{test_paras[0][0]}",
    'sub_path': False,
    'RRC_budget':RRC_budget
}

model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': 70,
    'test_batch_size': 1,
}



##########################################################################################
# main

def main_test(epoch,path,use_RRC=None,gen_type=None,cuda_device_num=None):
    if DEBUG_MODE:
        _set_debug_mode()
    if use_RRC:
        env_params['RRC_budget'] = RRC_budget
    else:
        env_params['RRC_budget'] = 0
    if gen_type is not None:
        env_params['tsplib_path'] = b + f"/data/{test_paras[gen_type][0]}"
        env_params['data_path'] =  b + f"/data/{test_paras[gen_type][0]}"
        tester_params['test_episodes'] = test_paras[gen_type][1]
        tester_params['test_batch_size'] = test_paras[gen_type][2]
    if cuda_device_num is not None:
        tester_params['cuda_device_num'] = cuda_device_num

    tester_params['model_load']={
        'path': path,
        'epoch': epoch,
    }

    logger_params = {
        'log_file': {
            'desc': f'test__tsplib_useRRC{use_RRC}_Gentype{gen_type}',
            'filename': 'log.txt'
        }
    }
    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)


    score_optimal, score_student, gap, result_dict, gaps_in_ranges = tester.run()
    return score_optimal, score_student,gap, result_dict, gaps_in_ranges



def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":

    path = model_load_path
    for i in [model_load_epoch]:
        # Run the main test process and collect results
        score_optimal, score_student, total_gap, result_dict, gaps_in_ranges = main_test(
            i, path, use_RRC=Use_RRC, gen_type=None, cuda_device_num=CUDA_DEVICE_NUM
        )

        # Construct the final output dictionary
        output = {
            "problems": result_dict,  # problem_name: {opt, score, gap}
            "mean_gap_per_range": {
                "0_200": float(gaps_in_ranges[0]) if len(gaps_in_ranges) > 0 else None,
                "200_500": float(gaps_in_ranges[1]) if len(gaps_in_ranges) > 1 else None,
                "500_1000": float(gaps_in_ranges[2]) if len(gaps_in_ranges) > 2 else None,
                "1000_5000": float(gaps_in_ranges[3]) if len(gaps_in_ranges) > 3 else None,
            },
            "global_mean_opt": float(score_optimal),
            "global_mean_score": float(score_student),
            "global_mean_gap": float(total_gap),
        }

        # Save all results as a JSON file
        with open(f'test_tsplib_statistics_useRRC{Use_RRC}_lehd.json', 'w') as f:
            json.dump(output, f, indent=2)