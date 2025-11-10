import sys
import torch
import itertools
import numpy as np
from logging import getLogger
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from POMO.TSPEnv import TSPEnv as Env
from POMO.TSPModel import TSPModel as Model
from POMO.utils import TimeEstimator,AverageMeter
from POMO.eval_longTrain import eval as EVAL

print("sys.path:", sys.path)  
class TSPTrainer:
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

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = trainer_params['result_folder']
        self.iteration_idx = trainer_params['idx_iteration']
        self.response_id = trainer_params['idx_response_id']
        # self.result_log = LogData()
        self.problem_type="min"
        
        self.use_early_stopping = False

        self.start_early_stopping = self.trainer_params['start_early_stopping']
        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        self.cuda_device_num = self.trainer_params['cuda_device_num']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

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
            if self.trainer_params['use_optimizer_state']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = self.start_epoch-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        
        # dataset path
        self.train_set_dir = env_params['train_root_dir']

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        print(f"module to evolve:{self.env_params['module_type']}")
        # Early Stopping Tracker
        best_aug_gap = float('inf') if self.problem_type == "min" else -float('inf')
        best_non_aug_gap = float('inf') if self.problem_type == "min" else -float('inf')
        no_improvement_counter = 0
        tolerance = self.trainer_params['tolerance']  # early stopping patience
        
        aug_gap_list, non_aug_gap_list = [],[]
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            if epoch>self.start_early_stopping and self.trainer_params['use_early_stopping']:
              self.use_early_stopping=True

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            # LR Decay
            self.scheduler.step()
        
            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            print("Epoch {:3d}/{:3d}: Epoch Train Score:{:.4f}, Epoch Train Loss:{:.4f}, Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], train_score, train_loss, elapsed_time_str, remain_time_str))
        

        
            # Final checkpoint if end or early stop
            if epoch == self.trainer_params['epochs'] or no_improvement_counter >= tolerance or (epoch % self.trainer_params['model_save_interval']==0 and epoch >= self.start_early_stopping):
                print("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-type{}-iteration{}-response{}-epoch{}.pt'.format(
                    self.result_folder, self.env_params['module_type'], self.iteration_idx, self.response_id, epoch))

                aug_gap, non_aug_gap = EVAL(val_root_dir=self.train_set_dir,
                                            checkpoint_folder=self.result_folder,
                                            idx_iteration=self.iteration_idx,
                                            idx_response_id=self.response_id,
                                            cuda_device_num=self.cuda_device_num,
                                            module_type=self.env_params['module_type'],
                                            epoch = epoch)
            # Early stopping logic
                is_improved = (aug_gap < best_aug_gap) if self.problem_type == "min" else (aug_gap > best_aug_gap)
                if self.use_early_stopping:
                    if is_improved:
                        best_aug_gap = aug_gap
                        no_improvement_counter = 0
                    else:
                        no_improvement_counter += 1
                        print(f"No improvement in train score for {no_improvement_counter} epoch(s).")
                else:
                    no_improvement_counter = 0
                    
                aug_gap_list.append(aug_gap)
                non_aug_gap_list.append(non_aug_gap)
                
                print(f'The aug gap and non aug gap for epoch{epoch} are {aug_gap} and {non_aug_gap} !')
                
            if no_improvement_counter >= tolerance and self.use_early_stopping:
                print(f"Early stopping triggered at epoch {epoch} due to no improvement in {tolerance} consecutive epochs.")
                print(f'The best gap remains {best_aug_gap} at epoch {epoch}!')
                break
            
        best_aug_gap, best_gap_idx = np.min(aug_gap_list), np.argmin(aug_gap_list)
        best_non_aug_gap = non_aug_gap_list[best_gap_idx]
        print(" *** Training Done *** ")
        return best_aug_gap, best_non_aug_gap

                


    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        # loop_cnt = 0
        while episode < train_num_episode:
            if episode % 1024 ==0:
              print(f"[DEBUG] --- Training episode {episode}/{train_num_episode}")
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size



        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
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
        return score_mean.item(), loss_mean.item()























