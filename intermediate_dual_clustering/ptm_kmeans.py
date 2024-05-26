from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import random
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from .trainer import train, valid
from .utils import model_select
from .dataset import make_dataset


class ptm_kmeans():
    """
    PTM - KMEANS
    2 단계 입니다.
    """
    def __init__(self, n_clusters, verbose):
        """
        n_clusters : 클러스터 수
        verbose : sib clustering 의 verbose
        """

        self.n_clusters = n_clusters
        self.kmeans_cluster = KMeans(n_clusters = self.n_clusters, 
                                     init = 'k-means++', 
                                     max_iter = 300, 
                                     random_state = 42)
        self.verbose = verbose
        self.ptm_kmeans_task = 'ptm_kmeans_task'

    def load_sentence_embed(self):
        """
        문장 임베딩을 생성합니다.
        라벨링이 안되어 있는 데이터 중 5퍼센트만 사용합니다.
        """

        random.seed(42)
        sample_num = int(len(self.unlabeled_target_list)/20)
        self.fraction_list = random.sample(self.unlabeled_target_list, sample_num)
        self.fraction_embeddings = self.embed_model.encode(self.fraction_list, show_progress_bar = self.verbose) # verbose

    def do_kmeans_clustering(self):
        """
        Kmeans clustering을 수행합니다.
        """

        self.kmeans_cluster.fit(self.fraction_embeddings)

        self.logger.info('[KMeans Clustering end!]')

        self.kmeans_labels = [int(label) for label in self.kmeans_cluster.labels_]
    
    def make_ptm_kmeans(self):
        """
        Kmeans clustering의 결과로 라벨링을 합니다.
        """
        
        ptm_kmeans_dict = dict()
        self.ptm_kmeans_list = list()

        for target, label in zip(self.fraction_list, self.kmeans_labels):
            ptm_kmeans_dict = {
                'target' : target,
                'label' : label
            }
            self.ptm_kmeans_list.append(ptm_kmeans_dict)

        return self.ptm_kmeans_list
    
    def ptm_kmeans_train_init(self):
        """
        ptm_kmeans_train을 하기전 ptm_sib_model 중 loss가 가장 낮은 model을 가져옵니다.
        optimizer, scheduler도 그에 맞게 init 해줍니다.
        """

        self.ptm_sib_model_path = model_select(self.ptm_sib_path)
        self.ptm_sib_model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(self.ptm_sib_model_path))
        self.ptm_sib_optimizer = torch.optim.Adam(self.ptm_sib_model.parameters(), lr = self.learning_rate)
        self.ptm_sib_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.ptm_sib_optimizer, T_max = 10, eta_min = 0)
    
    def ptm_kmeans_trainer(self,
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
        sib 모델에 kmeans 데이터를 기반으로 학습합니다.

        task_name : task 이름 입니다.
        epoch : ptm-kmeans-train의 epoch
        train_iter : train_dataloader
        valid_iter : valid_dataloader
        save_path : ptm-kmeans 모델에 저장 됩니다.
        criterion : criterion
        optimizer : optimizer
        scheduler : scheduler
        model : ptm-sib의 모델
        """                          
        
        for e in tqdm(range(epoch), desc = '[PTM - KMEANS]'):

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
            
        self.logger.info('[KMEANS TRAINING end!]')
        
    def ptm_kmeans_run(self):
        self.load_sentence_embed()
        self.do_kmeans_clustering()
        ptm_kemans_data = self.make_ptm_kmeans()
        ptm_kmeans_train_data, ptm_kmeans_valid_data = train_test_split(ptm_kemans_data, test_size = 0.2, random_state = 42)
        print(f"ptm_kmeans_train_data len : {len(ptm_kmeans_train_data)}")
        print(f"ptm_kmeans_valid_data len : {len(ptm_kmeans_valid_data)}")
        
        ptm_kmeans_train_data = make_dataset(ptm_kmeans_train_data, self.tokenizer, self.model_max_length) ##############
        ptm_kmeans_train_dataloader = DataLoader(ptm_kmeans_train_data,
                                              batch_size = self.batch_size,
                                              shuffle = True,
                                              collate_fn = ptm_kmeans_train_data.collate_fn
                                              )

        ptm_kmeans_valid_data = make_dataset(ptm_kmeans_valid_data, self.tokenizer, self.model_max_length) #####################
        ptm_kmeans_valid_dataloader = DataLoader(ptm_kmeans_valid_data,
                                              batch_size = self.batch_size,
                                              shuffle = True,
                                              collate_fn = ptm_kmeans_valid_data.collate_fn
                                              )
        
        self.ptm_kmeans_train_init()

        self.ptm_kmeans_trainer(task_name = self.ptm_kmeans_task,
                                epoch = self.epoch,
                                train_iter = ptm_kmeans_train_dataloader,
                                valid_iter = ptm_kmeans_valid_dataloader,
                                save_path = self.ptm_kmeans_path,
                                criterion = self.criterion,
                                optimizer = self.ptm_sib_optimizer,
                                scheduler = self.ptm_sib_scheduler,
                                model = self.ptm_sib_model
                             )
