
from logging import getLogger
import numpy as np
import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from LEHD_CVRP_p2.CVRP.VRPModel import VRPModel as Model
from LEHD_CVRP_p2.CVRP.test_setX import main_test
from LEHD_CVRP_p2.CVRP.VRPEnv import VRPEnv as Env
from LEHD_CVRP_p2.utils.utils import *

class VRPTrainer:
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
        self.result_folder = get_result_folder()
        # self.result_log = LogData()
        random_seed = 22
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        # cuda
        USE_CUDA = self.trainer_params['use_cuda'] # True
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num'] # 0
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params) # # {'problem_size': 100, 'pomo_size': 100}

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/p1-checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.start_epoch = 1 + model_load['epoch']
            # self.result_log.set_raw_data(checkpoint['result_log'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = 0
            self.logger.info('Saved Model Loaded !!')

        self.tolerance = trainer_params['tolerance']
        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        self.time_estimator.reset(self.start_epoch)
        
        self.env.load_raw_data(self.trainer_params['train_episodes'] )

        best_gap = None
        no_improvement_count = 0
        early_stopping_flag = False
        save_gap = []
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            if early_stopping_flag:
                self.logger.info(f"Early stopping triggered at epoch {epoch-1}. Best gap: {best_gap:.6f}")
                break
            
            self.logger.info('=================================================================')
            # self.env.shuffle_data()
            # Train
            train_score, train_student_score, train_loss = self._train_one_epoch(epoch)
            # self.result_log.append('train_score', epoch, train_score)
            # self.result_log.append('train_student_score', epoch, train_student_score)
            # self.result_log.append('train_loss', epoch, train_loss)
            # LR Decay
            self.scheduler.step()

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, self.trainer_params['epochs'],
                                                                                            elapsed_time_str, remain_time_str))
            # self.logger.info("Training Score:{:4f}, Training Student Score:{:4f}, Training Loss:{:4f}".format(train_score,
            #                                                                                                   train_student_score,train_loss))
            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            # img_save_interval = self.trainer_params['logging']['img_save_interval']

            # if epoch > 1:  # save latest images, every epoch
            #     self.logger.info("Saving log_image")
            #     image_prefix = '{}/latest'.format(self.result_folder)
                # util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],self.result_log, labels=['train_score'])
                # util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    # 'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/p2-checkpoint-{}.pt'.format(self.result_folder, epoch))

                score_optimal, score_student ,gap, p100_200, p200_500, p500_1000 = main_test(epoch, self.result_folder, use_RRC = False,
                                                              cuda_device_num=self.trainer_params['cuda_device_num'])

                save_gap.append([score_optimal, score_student,p100_200, p200_500, p500_1000, gap])
                np.savetxt(self.result_folder+'/gap.txt',save_gap,delimiter=',',fmt='%s')

            # if all_done or (epoch % img_save_interval) == 0:
                # image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                # util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],self.result_log, labels=['train_score'])
                # util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],self.result_log, labels=['train_loss'])
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    no_improvement_count = 0
                    early_stopping_flag = False
                    self.logger.info(f"Epoch {epoch}: gap improved to {gap:.6f}")
                else:
                    no_improvement_count += 1
                    self.logger.info(f"Epoch {epoch}: gap {gap:.6f} (no improvement, count={no_improvement_count})")
                    if no_improvement_count >= self.tolerance:
                        early_stopping_flag = True
                        self.logger.info(f"Early stopping will trigger after this epoch! (no improvement in 5 intervals)")
            if all_done:
                self.logger.info(" *** Training Done *** ")
                # self.logger.info("Now, printing log array...")
                # util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        
        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes'] # 70
        max_batch_size = self.trainer_params['max_batch_size'] # 4
        problem_indices = list(range(train_num_episode))
        np.random.shuffle(problem_indices)  # shuffle in every epoch

        for i, ep in enumerate(problem_indices):
            avg_score, score_student_mean, avg_loss, problem_batch_size = self._train_one_batch(ep, max_batch_size=max_batch_size)

            score_AM.update(avg_score, problem_batch_size)
            score_student_AM.update(score_student_mean, problem_batch_size)
            loss_AM.update(avg_loss, problem_batch_size)
            self.logger.info(
                f'Epoch {epoch:3d}: Train {i+1:3d}/{train_num_episode} ({100. * (i+1) / train_num_episode:1.1f}%)  '
                f'Score: {score_AM.avg:.4f}, Score_student: {score_student_AM.avg:.4f}, Loss: {loss_AM.avg:.4f}'
            )
        self.logger.info(
            f'Epoch {epoch:3d}: Train (100%)  Score: {score_AM.avg:.4f}, Score_student: {score_student_AM.avg:.4f}, Loss: {loss_AM.avg:.4f}'
        )
        return score_AM.avg, score_student_AM.avg, loss_AM.avg


    def _train_one_batch(self, episode,max_batch_size,epoch=None):

        self.model.train() # train status

        problem_batch_size, problem_size = self.env.load_problems(episode, max_batch_size = max_batch_size) # regenerate problems
        
        self.logger.info(f'Training problem X-n{problem_size} with batch size{problem_batch_size}...')
        reset_state, _, _ = self.env.reset(self.env_params['mode'])

        loss_list = []
        total_reward_teacher, total_reward_student = 0,0
        state, reward,reward_student, done = self.env.pre_step()

        current_step=0


        while not done:

            if current_step ==0:
                # print('current_step', current_step)
                selected_teacher = self.env.solution[:, 0, 0]
                selected_flag_teacher = self.env.solution[:, 0, 1]
                selected_student = selected_teacher
                selected_flag_student = selected_flag_teacher
                loss_mean = torch.tensor(0)

            else:

                loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                    self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                               raw_data_capacity=self.env.start_capacity_value)

                loss_mean = loss_node
                ################## perform Adam optimization through loss gradients
                self.model.zero_grad()
                loss_mean.backward()

                self.optimizer.step()

            current_step+=1
            state, reward, reward_student, done = self.env.step(selected_teacher, selected_student,selected_flag_teacher,selected_flag_student)  # Update the selected_teacher list and mask

            loss_list.append(loss_mean)
            
        if done:
            total_reward_teacher = reward
            total_reward_student = reward_student
            loss_mean = torch.tensor(loss_list).mean()

        return 0,0, loss_mean.item(), problem_batch_size
