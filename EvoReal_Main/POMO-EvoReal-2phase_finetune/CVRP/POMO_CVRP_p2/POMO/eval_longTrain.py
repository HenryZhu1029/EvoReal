##########################################################################################
# Import
import os
import sys
import re
import gc
import torch
import psutil
import json
import logging

from POMO.CVRPTester import CVRPTester as Tester

##########################################################################################
# Path Config


# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, "..")  # for problem_def
# sys.path.insert(0, "../..")  # for utils

#########################################################################################
# static parameters

# logger_params = {
#     'log_file': {
#         'desc': 'test_cvrp100',
#         'filename': 'log.txt'
#     }
# }

# logger = logging.getLogger('root')
# ##########################################################################################
# main

def main(epoch, checkpoint_path=None, cuda_device_num=0):
    
    DEBUG_MODE = False
    if torch.cuda.is_available():
        USE_CUDA = not DEBUG_MODE
    else:
        USE_CUDA = DEBUG_MODE

    CUDA_DEVICE_NUM = cuda_device_num
    logging.basicConfig(level=logging.INFO)
    
    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
    }


    tester_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'model_load': {
            'path': checkpoint_path,  # directory path of pre-trained model and log files saved.
            'epoch': epoch,  # epoch version of pre-trained model to laod.
        },
        'test_episodes': 1,
        'test_batch_size': 1,
        'augmentation_enable': True,
        'aug_factor': 8,
        'aug_batch_size': 1,
        'test_data_load': {
            'enable': True,
            'pt_path': "./dataset/train"
        },
    }
    root_dir = tester_params['test_data_load']['pt_path']
    if tester_params['augmentation_enable']:
        tester_params['test_batch_size'] = tester_params['aug_batch_size']
  
    pt_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.pt')])
    tester_logger_dict = {}
    # create logger
    # create_logger(**logger_params)
    # _print_config()
        
    for idx, pt_file in enumerate(pt_files):
      try:
        file_path = os.path.join(root_dir, pt_file)

        problem_name = os.path.splitext(pt_file)[0]
        match = re.search(r'n(\d+)', problem_name)
        if match:
            dimension = int(match.group(1)) - 1
        else:
            print("No dimension found!")
            dimension = 0
        # if dimension >= 120:
        #   continue
        env_params = {
            'problem_size': dimension,
            'pomo_size': dimension,
            'file_path': file_path,
            'problem_name': problem_name,
            'problem_idx': idx,
            'root_dir': root_dir
        }
        print(f"Start test {idx+1}/{len(pt_files)}: {pt_file}", flush=True)
        print(f"Running test for {file_path} with problem {problem_name}.")


        tester = Tester(env_params=env_params,
                        model_params=model_params,
                        tester_params=tester_params)


        score,aug_score = tester.run()

        if problem_name not in tester_logger_dict:
            tester_logger_dict[problem_name] = []

        tester_logger_dict[problem_name].append([score, aug_score])
        del tester  # delete tester
        torch.cuda.empty_cache()  # clean ram fractions
      except Exception as e:
          print(f"Error on {pt_file}: {e}")
    
    return tester_logger_dict

    # output_file = os.path.join(os.getcwd(), 'test_logger_dict.json')
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(tester_logger_dict, f, indent=4, ensure_ascii=False)
    # print(f"\n Test Results has been saved to: {output_file}")
    

# def _print_config():
#     logger = logging.getLogger('root')
#     logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
#     logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
#     [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

if __name__ == "__main__":
    import sys
    import json
    epoch = int(sys.argv[1])
    checkpoint_path = sys.argv[2]
    cuda_device_num = int(sys.argv[3])
    tester_logger_dict = main(epoch, checkpoint_path, cuda_device_num)
    # save to json
    output_path = f"tester_logger_dict_{epoch}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tester_logger_dict, f, indent=2, ensure_ascii=False)
    print(f"[Eval] Saved tester_logger_dict to {output_path}")