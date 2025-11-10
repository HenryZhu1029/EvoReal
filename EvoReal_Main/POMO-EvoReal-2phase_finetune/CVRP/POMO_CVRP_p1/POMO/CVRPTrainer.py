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
        
        # self.num_training_samples = env_params['num_total_samples']
        # self.train_root_dir = env_params['root_dir']
        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = './POMO/result/phase1_checkpoints'
        os.makedirs(self.result_folder,exist_ok=True)
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
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            #self.start_epoch = 1 + model_load['epoch']
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

        self.early_stop_flag = env_params['use_early_stopping']
        self.early_stopping_threshold = 0
        self.no_improvement_count = 0
        self.tolerance = env_params['tolerance']

    def _train_one_epoch(self, epoch):

        # train_dict = Shuffle_PTFilename_Loader(self.train_root_dir, shuffle=True)
        
        # statistics
        total_score = 0
        total_loss = 0

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        # for filename, dimension in train_dict.items():      
            # full_problem_path = os.path.join(self.train_root_dir, filename).replace('\\', '/')
            
        while episode < train_num_episode:
            if episode % 256 ==0:
              self.logger.info(f"[DEBUG] --- Training episode {episode}/{train_num_episode}")
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            
        # print(f"Begin Training Problem Size{dimension} with Batch Size{batch_size}...")
            avg_score, avg_loss = self._train_one_batch(batch_size)
            
            total_score += avg_score
            total_loss += avg_loss

            episode += batch_size
            
            if episode >= train_num_episode:
                epoch_avg_score = total_score/train_num_episode
                epoch_avg_loss = total_loss/train_num_episode
                self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                    .format(epoch, episode, train_num_episode, 100. * episode/train_num_episode,
                                            epoch_avg_score,epoch_avg_loss))
                self.logger.info("Training One Epoch is Done!")
            

        return epoch_avg_score, epoch_avg_loss

    def _train_one_batch(self, batch_size, problem_path=None):

        # Prep
        ###############################################
        self.model.train()
        # load training problems
        # self.env.use_saved_problems(problem_path, self.device, batch_size=batch_size, finetune_phase2=True)
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

        self.epoch_list = []
        self.score_list = []
        self.loss_list = []
        past_avg_aug_gap = []

        best_avg_gap, best_epoch = 0,0
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
                torch.save(checkpoint_dict, '{}/phase1-checkpoint-{}.pt'.format(self.result_folder, epoch))

                # compute avg AUG gap
                test_logger_dict = self._run_eval(epoch, self.cuda_device_num)
                avg_aug_gap = self.compute_avg_aug_gap(test_logger_dict)
                past_avg_aug_gap.append(avg_aug_gap)  
                self.logger.info(f"The average AUG gap of validation set on epoch {epoch} is {avg_aug_gap}!")  

                if len(past_avg_aug_gap) >= 2:
                    gap_delta = past_avg_aug_gap[-2] - past_avg_aug_gap[-1]
                    self.logger.info(f"[EarlyStop] Last checkpoint avg gap diff: {gap_delta:.6f}")
            
                # save best avg gap and epoch
                if best_epoch == 0 or avg_aug_gap < best_avg_gap:
                    best_avg_gap = avg_aug_gap
                    best_epoch = epoch
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                    self.logger.info(f'[EarlyStop] No improvement: best avg gap = {best_avg_gap:.4f}%, current = {avg_aug_gap:.4f}%. ({self.no_improvement_count}/{self.tolerance})')

            # ========== whether stop ==========
            if self.early_stop_flag and self.no_improvement_count>=self.tolerance:
                self.logger.info(f"[EarlyStop] Training stopped early due to no improvemnet for {self.tolerance} epochs!")
                self.logger.info(f'*********The best avg AUG gap found so far is {best_avg_gap}% at epoch {best_epoch}***********')
                break
            
        self.logger.info(" *** All training epochs is Done *** ")




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
            # 读取 json 文件
            os.makedirs('./eval_results',exist_ok=True)
            output_path = f"./eval_results/tester_logger_dict_{epoch}.json"
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
            self.logger.info("[Eval] No overlap between test_logger_dict and label_dict keys.")
        else:
            avg_gap = sum(gaps) / len(gaps)
            self.logger.info(f"[Eval] The average aug gap of {len(gaps)} problems is {avg_gap:.4f}%.")
        return avg_gap
        
        
        
        
        