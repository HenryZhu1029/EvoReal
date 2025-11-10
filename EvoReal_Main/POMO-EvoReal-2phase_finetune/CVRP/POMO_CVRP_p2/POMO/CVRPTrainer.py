import os
import subprocess
import json
import torch
from logging import getLogger

from POMO.CVRPEnv import CVRPEnv as Env
from POMO.CVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from POMO.utils import *
from POMO.CVRProblemDef import Shuffle_PTFilename_Loader

class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        
        self.num_training_samples = env_params['num_total_samples']
        # result folder, logger
        self.logger = getLogger(name='trainer')

        self.train_root_dir = env_params['root_dir']
        self.result_folder = './POMO/result/p2-checkpoints'
        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device
        
        self.cuda_device_num = cuda_device_num
        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/p1-checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            #self.start_epoch = 1 + model_load['epoch']

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = 0
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

        # for early stopping
        if self.env_params['use_early_stopping']:
            self.use_early_stopping = True
            self.early_stopping_threshold = self.env_params['early_stopping_threshold']
        else: 
            self.early_stopping_threshold = 0
            self.use_early_stopping = False
        self.tolerance = self.env_params['tolerance']

    def _train_one_epoch(self, epoch):

        loop_cnt = 0

        train_dict = Shuffle_PTFilename_Loader(self.train_root_dir, shuffle=True)
        
        # statistics
        total_score = 0
        total_loss = 0
        problem_trained = 0
        
        for filename, dimension in train_dict.items():
            
            full_problem_path = os.path.join(self.train_root_dir, filename).replace('\\', '/')

            batch_size = self.trainer_params['train_batch_size']

            if dimension <= 500:
                batch_size = batch_size
            elif dimension > 500 and dimension <= 650:
                batch_size = 2
            elif dimension >= 650 and dimension < 800:
                batch_size = 1
            else:
                continue
            
            print(f"Begin Training Problem Size{dimension} with Batch Size{batch_size}...")
            avg_score, avg_loss = self._train_one_batch(batch_size,problem_path=full_problem_path)
            
            total_score += avg_score
            total_loss += avg_loss
            problem_trained += 1
            loop_cnt += 1
            
            if loop_cnt <= self.num_training_samples:
                self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                    .format(epoch, problem_trained, self.num_training_samples, 100. * problem_trained/ self.num_training_samples,
                                            avg_score,avg_loss))

        
        if problem_trained == self.num_training_samples:
            print("Training One Epoch is Done!")
            
        epoch_avg_score = total_score / problem_trained
        epoch_avg_loss = total_loss / problem_trained
        return epoch_avg_score, epoch_avg_loss

    def _train_one_batch(self, batch_size, problem_path=None):

        # Prep
        ###############################################
        self.model.train()
        # load training problems
        self.env.use_saved_problems(problem_path, self.device, batch_size=batch_size, finetune_phase2=True)
        self.env.load_problems(batch_size)
        
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache() ##for gpu only
        return score_mean.item(), loss_mean.item()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        import numpy as np
        # import matplotlib.pyplot as plt
        # from IPython.display import clear_output,display
        # self.epoch_list = []
        # self.score_list = []
        # self.loss_list = []
        past_avg_aug_gap = []
        best_epoch = None
        best_gap = np.inf
        early_stop_flag = False
        no_improvement_count = 0
        # plt.ion()
        # self.fig_score, self.ax_score = plt.subplots()
        # self.fig_loss, self.ax_loss = plt.subplots()
        
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            
            self.logger.info('=================================================================')
            # Train
            epoch_avg_score, epoch_avg_loss = self._train_one_epoch(epoch)
            # LR Decay
            self.scheduler.step()

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),

                }
                torch.save(checkpoint_dict, '{}/p2-checkpoint-{}.pt'.format(self.result_folder, epoch))

                # compute avg AUG gap
                test_logger_dict = self._run_eval(epoch, self.cuda_device_num)
                avg_aug_gap = self.compute_avg_aug_gap(test_logger_dict)
                past_avg_aug_gap.append(avg_aug_gap)  
                print(f"The average AUG gap of validation set on epoch {epoch} is {avg_aug_gap}!")  
                # ========== early stopping ==========
                # if len(past_avg_aug_gap) >= 2:
                #     gap_delta = abs(past_avg_aug_gap[-2] - past_avg_aug_gap[-1])
                #     print(f"[EarlyStop] Last checkpoint avg gap diff: {gap_delta:.6f}")
                #     if gap_delta <= self.early_stopping_threshold:
                #         print(f"[EarlyStop] Early stopping triggered: avg gap change {gap_delta:.6f} <= threshold {self.early_stopping_threshold}")
                #         early_stop_flag = True
                if self.use_early_stopping:
                    if best_epoch is None or avg_aug_gap < best_gap:
                        best_gap =avg_aug_gap
                        best_epoch = epoch
                        no_improvement_count = 0
                    else:
                        no_improvement_count +=1
                    
                    if no_improvement_count>=self.tolerance:
                        early_stop_flag = True
                        
                    
                         

            # self.epoch_list.append(epoch)
            # self.score_list.append(epoch_avg_score)
            # self.loss_list.append(epoch_avg_loss)

            # clear_output(wait=True)
            # self.ax_score.clear()
            # self.ax_score.plot(self.epoch_list, self.score_list, marker='o')
            # self.ax_score.set_title("Average Score per Epoch")
            # self.ax_score.set_xlabel("Epoch")
            # self.ax_score.set_ylabel("Score")
            # display(self.fig_score)

            # self.ax_loss.clear()
            # self.ax_loss.plot(self.epoch_list, self.loss_list, marker='x')
            # self.ax_loss.set_title("Average Loss per Epoch")
            # self.ax_loss.set_xlabel("Epoch")
            # self.ax_loss.set_ylabel("Loss")
            # display(self.fig_loss)
            
            # ========== whether stop ==========
            if early_stop_flag:
                print("[EarlyStop] Training stopped early due to small gap change.")
                break
            
        self.logger.info(" *** All training epochs is Done *** ")

        score_plot_path = os.path.join(self.result_folder, "epoch_score_curve.png")
        loss_plot_path = os.path.join(self.result_folder, "epoch_loss_curve.png")

        self.fig_score.savefig(score_plot_path)
        self.fig_loss.savefig(loss_plot_path)

        self.logger.info(f"Score plot saved to {score_plot_path}")
        self.logger.info(f"Loss plot saved to {loss_plot_path}")
        

    def _run_eval(self, epoch, cuda_device_num=0):
            eval_script = os.path.join("POMO", "eval_longTrain.py")
            checkpoint_path = self.result_folder
            cmd = [
                "python", eval_script,
                str(epoch),
                checkpoint_path,
                str(cuda_device_num)
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd() 
            print(f"\n[Eval] Running eval: {' '.join(cmd)}")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,env=env)
            for line in process.stdout:
                print(line, end='')
            process.wait()

            output_path = f"tester_logger_dict_{epoch}.json"
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    tester_logger_dict = json.load(f)
            else:
                tester_logger_dict = None
                print(f"[Warning] tester_logger_dict file not found: {output_path}")
            print(f"[Eval] Eval for epoch {epoch} finished with exit code {process.returncode}")
            return tester_logger_dict

    def compute_avg_aug_gap(self, test_logger_dict, label_file='./dataset/cvrp_label.json'):
        with open(label_file, 'r') as f:
            label_dict = json.load(f)

        gaps = []
        for prob, scores in test_logger_dict.items():
            if prob not in label_dict:
                continue
            aug_score = scores[0][1]
            opt = label_dict[prob]["normalized_label"]
            gap = (aug_score - opt) / opt * 100
            gaps.append(gap)

        if len(gaps) == 0:
            avg_gap = None
            print("[Eval] No overlap between test_logger_dict and label_dict keys.")
        else:
            avg_gap = sum(gaps) / len(gaps)
            print(f"[Eval] The average aug gap of {len(gaps)} problems is {avg_gap:.4f}%.")
        return avg_gap