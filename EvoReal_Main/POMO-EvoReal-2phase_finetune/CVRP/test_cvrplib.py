import re
import sys
import gc
import torch
import psutil
import json
import os
import logging


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Add POMO_CVRP_p2 to sys.path so 'from POMO.CVRPTester import ...' works
sys.path.insert(0, os.path.join(BASE_DIR, "POMO_CVRP_p2"))
from POMO_CVRP_p2.POMO.utils import create_logger,get_num_samples
from POMO_CVRP_p2.POMO.CVRPTester import CVRPTester as Tester

DEBUG_MODE = False
if torch.cuda.is_available():
    USE_CUDA = not DEBUG_MODE
else:
    USE_CUDA = DEBUG_MODE

CUDA_DEVICE_NUM = 0

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
        'path': './POMO_CVRP_p2/POMO/result/saved_p2_checkpoints',  # directory path of pre-trained model and log files saved.
        'epoch': 60,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 1,
    'test_batch_size': 1,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 1,
    'test_data_load': {
        'enable': True,
        'json_path': "./POMO_CVRP_p2/dataset/all_setX",
        'pt_path': "./POMO_CVRP_p2/dataset/setX_pt"

    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_cvrp100',
        'filename': 'log.txt'
    }
}

logger = logging.getLogger('root')
##########################################################################################

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

##########################################################################################

# problem dir setup
root_dir = tester_params['test_data_load']['pt_path']

# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()
    logging.basicConfig(level=logging.INFO)
    pt_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.pt')])
    tester_logger_dict = {}
    # create logger
    create_logger(**logger_params)
    _print_config()
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
        logger.info(f"Running test for {file_path} with problem {problem_name}.")


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


    # output_file = os.path.join(os.getcwd(), 'test_logger_dict.json')
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(tester_logger_dict, f, indent=4, ensure_ascii=False)
    # print(f"\n Test Results has been saved to: {output_file}")

    return tester_logger_dict

def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def compute_and_save_statistics(
    logger_path='./test_logger_dict.json',
    label_path='./cvrp_labels.json',
    save_path='./test_cvrplib_statistics_pomo.json'
):
    # Load predicted results and label information
    with open(logger_path, 'r') as f:
        logger_dict = json.load(f)
    with open(label_path, 'r') as f:
        label_dict = json.load(f)

    # Define intervals for statistics
    intervals = [(0, 200), (200, 500), (500, 1002)]
    interval_names = ['[0,200)', '[200,500)', '[500,1002)']
    gap_by_interval = {name: [] for name in interval_names}
    all_aug_gaps = []
    all_non_aug_gaps = []

    per_problem_results = {}

    for prob, scores in logger_dict.items():
        if prob not in label_dict:
            continue

        aug_score_norm = scores[0][1]
        non_aug_score_norm = scores[0][0]
        opt_norm = label_dict[prob]["normalized_label"]
        opt_real = label_dict[prob]["UB/opt"]  # real optimal value

        if opt_norm == 0:
            continue  # avoid division by zero

        # Compute gap using normalized scores and labels (in %)
        aug_gap = (aug_score_norm - opt_norm) / opt_norm * 100
        non_aug_gap = (non_aug_score_norm - opt_norm) / opt_norm * 100

        all_aug_gaps.append(aug_gap)
        all_non_aug_gaps.append(non_aug_gap)

        num_customers = label_dict[prob]["num_of_customers"]
        for (low, high), name in zip(intervals, interval_names):
            if low <= num_customers < high:
                gap_by_interval[name].append((aug_gap, non_aug_gap))

        # Restore original scale scores for reference
        scale = opt_real / opt_norm
        aug_score_real = aug_score_norm * scale
        non_aug_score_real = non_aug_score_norm * scale

        # Only save real opt/scores and normalized gaps
        per_problem_results[prob] = {
            "num_customers": num_customers,
            "Opt": opt_real,   # optimum value from official cvrplib (setX) dataset
            "aug_score_real": aug_score_real,
            "non_aug_score_real": non_aug_score_real,
            "aug_gap": aug_gap,
            "non_aug_gap": non_aug_gap
        }

    # Compute interval and total statistics
    statistics = {}
    statistics['total'] = {
        "avg_aug_gap": sum(all_aug_gaps) / len(all_aug_gaps) if all_aug_gaps else float('nan'),
        "avg_non_aug_gap": sum(all_non_aug_gaps) / len(all_non_aug_gaps) if all_non_aug_gaps else float('nan'),
        "problem_count": len(all_aug_gaps)
    }
    for name in interval_names:
        gap_list = gap_by_interval[name]
        if not gap_list:
            statistics[name] = {"note": "no problems in this interval"}
            continue
        aug_gaps = [x[0] for x in gap_list]
        non_aug_gaps = [x[1] for x in gap_list]
        statistics[name] = {
            "avg_aug_gap": sum(aug_gaps) / len(aug_gaps),
            "avg_non_aug_gap": sum(non_aug_gaps) / len(non_aug_gaps),
            "problem_count": len(gap_list)
        }

    # Combine and save final results
    final_output = {
        "per_problem": per_problem_results,
        "statistics": statistics
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"Gap details and statistics saved to: {save_path}")

    # Optional: pretty-print statistics in console
    print(json.dumps(statistics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    tester_logger_dict = main()  # run test and get predictions
    compute_and_save_statistics()

