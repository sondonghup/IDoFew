import os
import pickle
from sib import SIB
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import time
import pandas as pd

from .dataset import make_dataset
from .trainer import train, valid
from .log_tf_idf import get_log_tf_idf


class ptm_sib():
    """
    PTM - SIB
    1 단계 입니다.
    """
    def __init__(self, n_clusters, verbose):
        """
        n_clusters : 클러스터 수
        verbose : sib clustering 의 verbose
        """

        self.n_clusters = n_clusters
        self.sib_cluster = SIB(n_clusters = self.n_clusters,
                               random_state = 42,
                               n_init = 10,
                               n_jobs = -1,
                               max_iter = 15,
                               verbose = verbose)
        self.log_tf_idf_path = '../IDoFew/cache'
        self.ptm_sib_task = 'ptm_sib_task'

    def load_log_tf_idf(self):
        """
        라벨링이 안되어 있는 데이터를 기준으로 log_tf_idf를 생성합니다.
        이미 생성되어 있으면 ./cache에서 가져옵니다.
        """
        start_log_tf_idf = time.time()
        
        if not os.path.exists(self.log_tf_idf_path):
            os.makedirs(self.log_tf_idf_path)

        if os.path.exists(f"{self.log_tf_idf_path}/log_tf_idf.pickle"):
            with open(f"{self.log_tf_idf_path}/log_tf_idf.pickle", 'rb')as f:
                self.log_tf_idf = pickle.load(f)
        else :
            self.log_tf_idf = get_log_tf_idf(self.unlabeled_target_list)
            with open(f"{self.log_tf_idf_path}/log_tf_idf.pickle", 'wb')as f:
                pickle.dump(self.log_tf_idf, f)

        end_log_tf_idf = time.time()
        
        self.logger.info(f"[Log tf idf end!] time : {end_log_tf_idf - start_log_tf_idf}")

    def do_sib_clustering(self):
        """
        sib clustering을 수행합니다.
        """

        start_sib_cluster = time.time()
        
        self.sib_cluster.fit(self.log_tf_idf)

        end_sib_cluster = time.time()
        
        self.logger.info(f"[SIB Clustering end!] time : {end_sib_cluster - start_sib_cluster}")

        self.sib_labels = [int(label) for label in self.sib_cluster.labels_]
    
    def make_ptm_sib(self):
        """
        sib clustering의 결과로 라벨링을 합니다.
        """

        ptm_sib_dict = dict()
        self.ptm_sib_list = list()

        for target, label in zip(self.unlabeled_target_list, self.sib_labels):
            ptm_sib_dict = {
                'target' : target,
                'label' : label
            }
            self.ptm_sib_list.append(ptm_sib_dict)

        if not os.path.exists(self.save_sib_path):
            os.makedirs(self.save_sib_path)
            
        pd.DataFrame(self.ptm_sib_list).to_csv(self.save_sib_path + wandb.run.name + "_sib.csv", index = False)

        return self.ptm_sib_list
    
    def ptm_sib_trainer(self,
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
        sib 데이터를 기반으로 학습합니다.

        task_name : task 이름 입니다.
        epoch : ptm-sib-train의 epoch
        train_iter : train_dataloader
        valid_iter : valid_dataloader
        save_path : ptm-sib 모델에 저장 됩니다.
        criterion : criterion
        optimizer : optimizer
        scheduler : scheduler
        model : 사전 학습 모델
        """

        for e in tqdm(range(epoch), desc = '[PTM - SIB]'):

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
            
        self.logger.info('[SIB TRAINING end!]')


    def ptm_sib_run(self):
        self.load_log_tf_idf()
        self.do_sib_clustering()
        ptm_sib_data = self.make_ptm_sib()
        ptm_sib_train_data, ptm_sib_valid_data = train_test_split(ptm_sib_data, test_size = 0.2, random_state = 42)
        print(f"ptm_sib_train_data len : {len(ptm_sib_train_data)}")
        print(f"ptm_sib_valid_data len : {len(ptm_sib_valid_data)}")
        
        ptm_sib_train_data = make_dataset(ptm_sib_train_data, self.tokenizer, self.model_max_length)
        ptm_sib_train_dataloader = DataLoader(ptm_sib_train_data,
                                              batch_size = self.batch_size,
                                              shuffle = True,
                                              collate_fn = ptm_sib_train_data.collate_fn
                                              )

        ptm_sib_valid_data = make_dataset(ptm_sib_valid_data, self.tokenizer, self.model_max_length)
        ptm_sib_valid_dataloader = DataLoader(ptm_sib_valid_data,
                                              batch_size = self.batch_size,
                                              shuffle = True,
                                              collate_fn = ptm_sib_valid_data.collate_fn
                                              )

        self.ptm_sib_trainer(task_name = self.ptm_sib_task,
                             epoch = self.epoch,
                             train_iter = ptm_sib_train_dataloader,
                             valid_iter = ptm_sib_valid_dataloader,
                             save_path = self.ptm_sib_path,
                             criterion = self.criterion,
                             optimizer = self.optimizer,
                             scheduler = self.scheduler,
                             model = self.pretrained_model
                             )
