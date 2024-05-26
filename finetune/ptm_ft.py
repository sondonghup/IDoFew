import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
import wandb

from intermediate_dual_clustering.trainer import train, valid
from intermediate_dual_clustering.dataset import make_dataset
from intermediate_dual_clustering.utils import model_select

class ptm_ft():
    """
    PTM - FT
    3 단계 입니다.
    """
    def __init__(self, n_clusters):
        """
        n_clusters : 클러스터 수 = 분류기의 카테고리 수 
        """
        
        self.n_clusters = n_clusters
        self.ptm_ft_task = 'ptm_ft_task'

    def ptm_ft_train_init(self):
        """
        ptm_ft_train을 하기전 ptm_kmeans_model 중 loss가 가장 낮은 model을 가져옵니다.
        optimizer, scheduler도 그에 맞게 init 해줍니다.
        """

        self.ptm_kmeans_model_path = model_select(self.ptm_kmeans_path)
        self.ptm_kmeans_model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(self.ptm_kmeans_model_path, num_labels = self.n_clusters))
        self.ptm_kmeans_optimizer = torch.optim.Adam(self.ptm_kmeans_model.parameters(), lr = self.learning_rate)
        self.ptm_kmeans_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.ptm_kmeans_optimizer, T_max = 10, eta_min = 0)

    def ptm_ft_trainer(self,
                       task_name,
                       epoch,
                       train_iter,
                       valid_iter,
                       save_path,
                       criterion,
                       optimizer,
                       scheduler,
                       model
                       ):
        """
        kmeans 모델에 라벨링 된 데이터 기반으로 학습합니다.

        task_name : task 이름입니다.
        epoch : ptm-ft-train의 epoch
        train_iter : train_dataloader
        valid_iter : valid_dataloader
        save_path : ptm-ft 모델에 저장 됩니다.
        criterion : criterion
        optimizer : optimizer
        scheduler : scheduler
        model : ptm-kmeans의 모델
        """  
        
        for e in tqdm(range(epoch), desc = '[PTM - FT]'):

            train(task_name,
                  train_iter,
                  criterion,
                  optimizer,
                  scheduler,
                  model
                  )
            
            valid(task_name,
                  valid_iter,
                  criterion,
                  model,
                  save_path
                  )
            
        self.logger.info('[FINETUNING end!]')

    def ptm_ft_run(self):

        ptm_ft_train_data, ptm_ft_valid_data = train_test_split(self.labeled_target_list, test_size = 0.2, random_state = 42)

        ptm_ft_train_data = make_dataset(ptm_ft_train_data, self.tokenizer, self.model_max_length)
        ptm_ft_train_dataloader = DataLoader(ptm_ft_train_data,
                                             batch_size = self.batch_size,
                                             shuffle = True,
                                             collate_fn = ptm_ft_train_data.collate_fn
                                             )
        
        ptm_ft_valid_data = make_dataset(ptm_ft_valid_data, self.tokenizer, self.model_max_length)
        ptm_ft_valid_dataloader = DataLoader(ptm_ft_valid_data,
                                             batch_size = self.batch_size,
                                             shuffle = True,
                                             collate_fn = ptm_ft_valid_data.collate_fn
                                             )
        
        self.ptm_ft_train_init()

        self.ptm_ft_trainer(task_name = self.ptm_ft_task,
                             epoch = self.epoch,
                             train_iter = ptm_ft_train_dataloader,
                             valid_iter = ptm_ft_valid_dataloader,
                             save_path = self.ptm_ft_path,
                             criterion = self.criterion,
                             optimizer = self.ptm_kmeans_optimizer,
                             scheduler = self.ptm_kmeans_scheduler,
                             model = self.ptm_kmeans_model
                             )
