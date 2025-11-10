import sys
import torch
import os
import numpy as np
from datetime import datetime
from logging import getLogger
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from POMO.TSPEnv import TSPEnv as Env
from POMO.TSPModel import TSPModel as Model
from POMO.utils import TimeEstimator,AverageMeter
from POMO.Problem_Def import Shuffle_Filename_Loader
from Test_TSPLib95 import main_test
from POMO.tspdata_loader import load_labels

class TSPTrainer:
    def __init__(self,
                 global_params,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.global_params = global_params
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        
        self.num_training_samples = env_params['num_total_samples']
        self.train_root_dir = env_params['root_dir']
        # result folder, logger
        self.logger = getLogger(name='trainer')
        time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.result_folder = f'./POMO/result/p2-checkpoints-{time_stamp}'
        os.makedirs(self.result_folder,exist_ok=True)
        self.labels_dict = load_labels(root_dir='./')
        
        # seed
        random_seed = 2
        torch.manual_seed(random_seed)
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

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = 0
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        
        self.use_early_stopping = trainer_params['use_early_stopping']
        self.early_stopping_flag = False
        if self.use_early_stopping:
            self.stop_threshold = trainer_params['stop_threshold']
            self.tolerance = trainer_params['tolerance']
        else:
            self.tolerance = np.inf
    
    def _train_one_epoch(self, epoch, train_root_dir):
        
        loop_cnt = 0
        problem_trained = 0
        train_dict = Shuffle_Filename_Loader(train_root_dir, shuffle=True)
        
        total_score = 0
        total_loss = 0
        for filename, dimension in train_dict.items():
            full_problem_path = os.path.join(train_root_dir, filename).replace('\\', '/')
            pomo_size = dimension
            batch_size = self.trainer_params['train_batch_size']
            if pomo_size<500:
                batch_size = batch_size
            elif pomo_size >=500 and pomo_size<750:
                batch_size = 2
            elif pomo_size>=750 and pomo_size<1002:
                batch_size = 1
            else:
                continue
            print(f"Begin Training Problem Size{dimension} with Batch Size{batch_size}...")
            avg_score, avg_loss = self._train_one_batch(full_problem_path, batch_size, pomo_size)
            
            total_score += avg_score
            total_loss += avg_loss
            problem_trained += 1
            loop_cnt += 1

            if loop_cnt <= self.num_training_samples:
                print('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'.format(
                epoch, problem_trained, self.num_training_samples,
                100. * problem_trained / self.num_training_samples, avg_score, avg_loss))

        if problem_trained == self.num_training_samples:
            print("Training One Epoch is Done!")
        epoch_avg_score = total_score / problem_trained
        epoch_avg_loss = total_loss / problem_trained
        return epoch_avg_score, epoch_avg_loss

            
            
    def _train_one_batch(self,full_problem_path,batch_size,pomo_size):
        
        
        
        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(full_problem_path,batch_size,pomo_size)          
        reset_state, _, _ = self.env.reset()            
        self.model.pre_forward(reset_state)                   

        prob_list = torch.zeros(size=(batch_size, pomo_size, 0))
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
        # prob_list = torch.clamp(prob_list, min=1e-15)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage *log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        self.model.zero_grad()
        loss_mean.backward()
    

        self.optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache() ##for gpu only
        return score_mean.item(), loss_mean.item()
    
    def run(self):
        self.time_estimator.reset(self.start_epoch)
        # import matplotlib.pyplot as plt
        # from IPython.display import clear_output,display
        self.epoch_avg_gap_list = []
        add_count = 0
        best_epoch = 0
        best_gap = None
        no_improvement_count = 0
        # self.score_list = []
        # self.loss_list = []
        # plt.ion()
        # self.fig_score, self.ax_score = plt.subplots()
        # self.fig_loss, self.ax_loss = plt.subplots()
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            # Train

            avg_score, avg_loss = self._train_one_epoch(epoch, self.train_root_dir)
           

            self.scheduler.step()
            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}, Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.global_params['total_epochs'], elapsed_time_str,remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']

           
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                    
                }
                torch.save(checkpoint_dict, '{}/p2-checkpoint-{}.pt'.format(self.result_folder,epoch))

                tester_logger_dict = main_test(problem_dir=self.env_params['val_root_dir'],checkpoint_path=self.result_folder,epoch=epoch)
                
                epoch_avg_gap = self.calc_avg_gap(tester_logger_dict=tester_logger_dict,label_dict=self.labels_dict)

                if epoch_avg_gap is not None:
                    self.epoch_avg_gap_list.append(epoch_avg_gap)
                    add_count+=1
                    if best_gap is None or best_gap>epoch_avg_gap:
                        best_gap = epoch_avg_gap
                        best_epoch = epoch
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    
                if add_count>=2 and no_improvement_count >= self.tolerance:
                    self.early_stopping_flag = True
                else:
                    self.early_stopping_flag = False
                    
            if self.early_stopping_flag:
                self.logger.info(f'Early stopping triggered. The best avg AUG gap on the whole validation set is {best_gap} at epoch{best_epoch}.')       
                break
 
                    
                    


    def calc_avg_gap(self, tester_logger_dict, label_dict=None):
        gaps = []
        for name, results in tester_logger_dict.items():
            if name not in label_dict:
                continue
            #  results=[ [score, aug_score] ]
            aug_score = results[0][1]
            opt = label_dict[name]["opt"]
            gap = (aug_score - opt) / opt * 100
            gaps.append(gap)
        avg_gap = sum(gaps) / len(gaps) if gaps else None
        return avg_gap





















