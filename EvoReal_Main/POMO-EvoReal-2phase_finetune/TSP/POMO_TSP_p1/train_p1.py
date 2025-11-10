# -*- coding: utf-8 -*-
import os
import json
import logging
import numpy as np
import time
from datetime import datetime
from pathlib import Path

from POMO.utils import *
from POMO.train_longTrain import train as TRAIN
from POMO.eval_longTrain import eval as EVAL


# Set log output
logging.basicConfig(level=logging.INFO)
CUDA_DEVICE_NUM = 0
logger = logging.getLogger(__name__)

def main():
    start_time = time.time()
    # === Set the project path to "./tspdata_evo_main" ===
    ROOT_DIR = Path(__file__).resolve().parent
    os.chdir(ROOT_DIR)
    
    # experiment id
    idx_iteration = 0
    idx_response_id = datetime.now().strftime('%Y%m%d')
    # === path config ===
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M')
    checkpoint_folder = f"POMO/result/checkpoints/checkpoint_{time_stamp}"
    os.makedirs(checkpoint_folder,exist_ok=True)
    train_set_dir = "dataset/Gen_Train"
    stdout_file_path = f"stdout_test{idx_response_id}.json"

    # training hyperparameters
    tolerance = 100 # set a high tolerance (>=num_outer_epochs) to avoid early-stopping
    model_save_interval = 5 # evaluate the model at every 5 epochs
    eval_start_epoch = 1 # start evaluation at epoch 1*5 = 5 
    num_outer_epochs = 61  # number of outer epoch, finetuning the model for a total of (61-1) * 5 = 300 epochs 


    # === Perform multiple TRAIN + EVAL ===
    results_per_epoch = {}
    patience_counter = 0
    best_score = float('inf')  # Lower score is better

    with open(stdout_file_path, "w") as f:
        try:
            
            # generation weights are set according to the division of S1,S2,S3 used in EvoReal generator evolution.
            gen_weights = [19,12,17] # generating raito of [S1, S2, S3] tsplibs samples, [19,12,17] is the ratio of rule-based-sclustering, [15,16,17] is the ratio of spectral-clusterings

            for epoch in range(1, num_outer_epochs):  
            
                logger.info(f"[TEST] Running TRAIN for outer epoch {epoch}...")
                TRAIN(checkpoint_folder, idx_iteration, idx_response_id, CUDA_DEVICE_NUM, module_type="mixed", epoch=epoch, ratio=gen_weights,model_save_interval = model_save_interval)
                
                if epoch>=eval_start_epoch:
                    logger.info(f"[TEST] Running EVAL for outer epoch {epoch}...")
                    result = EVAL(train_set_dir, checkpoint_folder, idx_iteration, idx_response_id, CUDA_DEVICE_NUM, module_type="mixed", epoch=epoch)
                    print("==== Raw Eval Result Sample ====")
                    # print(result)  # 来自 eval_longTrain.py 返回的字典

                    print("==== Label Dict Sample ====")

                    label_dict = get_label()
                    # print(label_dict)

                    # print("==== Computed Gap Dict ====")
                    # print(type_gap_dict)
                    

                    type_gap_dict = compute_aug_gap_by_size(result, label_dict)

                    # extract the aug gaps for each type
                    # print("==== Computed Gap Dict ====")
                    # print(type_gap_dict)

                    # 提取各区间gap
                    gap_0_200 = type_gap_dict.get("[0,200)", 0.0)
                    gap_200_500 = type_gap_dict.get("[200,500)", 0.0)
                    gap_500_1000 = type_gap_dict.get("[500,1000)", 0.0)
                    gap_1000_5000 = type_gap_dict.get("[1000,5000)", 0.0)
                    overall_gap = type_gap_dict.get("Overall", 0.0)

                    result_summary = {
                        "[0,200)": round(gap_0_200, 6),
                        "[200,500)": round(gap_200_500, 6),
                        "[500,1000)": round(gap_500_1000, 6),
                        "[1000,5000)": round(gap_1000_5000, 6),
                        "Overall": round(overall_gap, 6)
                    }
                    logger.info(f"[TEST] Epoch {epoch} - Gaps by size: {result_summary}")
                    results_per_epoch[f"epoch_{epoch}"] = result_summary

                    current_score = overall_gap  # 以总体平均gap作为early-stopping指标


                    # Early stopping check
                    if current_score < best_score:
                        best_score = current_score
                        logger.info(f"[TEST] Current Best Score:{best_score:.6f}")
                        patience_counter = 0  # Reset patience if improved
                    else:
                        patience_counter += 1
                        logger.info(f"[TEST] No improvement. Patience counter: {patience_counter}")

                    if patience_counter >= tolerance:
                        logger.info(f"[TEST] Early stopping triggered at epoch {epoch}. Best score: {best_score:.6f}")
                        break

            json.dump(results_per_epoch, f, indent=2)


            # Count Total Training Time
            end_time = time.time()
            total_time_sec = end_time - start_time
            hours, rem = divmod(total_time_sec, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            logger.info(f"[TEST] Total execution time: {time_str}")

        except Exception as e:
            f.write("Traceback: " + str(e) + '\n')
            logging.error(f"[TEST] Execution failed: {e}")

if __name__ == "__main__":
    main()

