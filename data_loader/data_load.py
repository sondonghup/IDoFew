import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import sentencepiece as spm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
import torch
import wandb

from intermediate_dual_clustering.ptm_kmeans import ptm_kmeans
from intermediate_dual_clustering.ptm_sib import ptm_sib
from intermediate_dual_clustering.utils import set_logger
from finetune.ptm_ft import ptm_ft
from make_tokenizer.pecab_tokenizer import make_tokenizer


class data_load_manager(ptm_kmeans, ptm_sib, ptm_ft):
    def __init__(self,
                 unlabeled_data_path,
                 labeled_data_path,
                 n_clusters,
                 embed_model_path,
                 verbose,
                 tokenizer_path,
                 vocab_size,
                 use_hf_tokenizer,
                 pretrained_model_path,
                 batch_size,
                 epoch,
                 learning_rate,
                 save_path,
                 hf_tokenizer_path = ""):
        """
        unlabeled_data_path : 라벨링이 되지 않은 데이터 경로
        labeled_data_path : 라벨링이 되어 있는 데이터 경로
        n_clusters : 클러스터 개수 -> 최종 분류기의 분류 개수가 됨
        embed_model_path : ptm - kmeans에 사용되는 임베딩 모델
        verbose : sib 클러스터링과 임베딩에 대한 verbose
        tokenizer_path : 토크나이저가 생성될 경로 (토크나이저를 직접 생성할 경우)
        vocab_size : 토크나이저 단어 수
        use_hf_tokenizer : 허깅페이스 토크나이저 (토크나이저를 생성하지 않고 허깅페이스의 토크나이저를 사용할 경우)
        pretrained_model_path : text classification에 사용될 모델
        batch_size : text classification의 batch_size
        epoch : text classification의 epoch
        learning_rate : text classification 의 learning_rate
        save_path : 각 단계에서 학습된 모델들이 저장될 경로

        토크나이저를 직접 생성할 경우 pecab을 사용 하기 때문에 만약 영어 데이터로 실행을 한다면 hf 토크나이저를 사용해주세요
        """
        
        wandb.init(project = f"IDoFew Text classification")
        
        self.unlabeled_data_path = unlabeled_data_path
        self.labeled_data_path = labeled_data_path
        self.tokenizer_path = tokenizer_path
        self.embed_model = SentenceTransformer(embed_model_path)
        self.use_hf_tokenizer = use_hf_tokenizer
        self.hf_tokenizer_path = hf_tokenizer_path
        self.pretrained_model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, num_labels = n_clusters))
        self.model_max_length = self.pretrained_model.module.config.max_position_embeddings
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.pretrained_model.parameters(), lr = learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = 10, eta_min = 0)
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.ptm_sib_path = f"../IDoFew_data/ptm_sib/{save_path}{wandb.run.name}/" # 1단계인 ptm-sib 의 모델이 저장 될 경로
        self.ptm_kmeans_path = f"../IDoFew_data/ptm_kmeans/{save_path}{wandb.run.name}/" # 2단계인 ptm-kmeans 의 모델이 저장 될 경로
        self.ptm_ft_path = f"../IDoFew_data/ptm_ft/{save_path}{wandb.run.name}/" # 3단계인 ptm-ft 의 모델이 저장 될 경로
        self.log_dir = f"../IDoFew_data/log/{wandb.run.name}/"
        self.save_sib_path = '../IDoFew_data/sib/'

        self.vocab_size = vocab_size

        self.logger = set_logger(self.log_dir)

        
        
        ptm_sib.__init__(self, n_clusters, verbose)
        ptm_kmeans.__init__(self, n_clusters, verbose)
        ptm_ft.__init__(self, n_clusters)

    def load_unlabeled_data(self):
        """
        라벨링이 되지 않은 데이터를 로드 합니다.
        """

        if self.unlabeled_data_path.endswith('.csv'):
            data = pd.read_csv(self.unlabeled_data_path)
        
        unique_columns_list = list(data.columns.unique())

        unlabeled_target_list = list()

        # instruction, output, input이 column으로 주어질때
        if ('instruction' in unique_columns_list) and ('output' in unique_columns_list) and ('input' in unique_columns_list):
            for instruction, output, input in zip(data['instruction'], data['output'], data['input']):
                instruction = f"{instruction}\n{input}"
                target_data = f"instruction : {instruction}\noutput : {output}"
                unlabeled_target_list.append(target_data)
            self.unlabeled_target_list = unlabeled_target_list
        
        # instruction, output이 column으로 주어질때
        elif ('instruction' in unique_columns_list) and ('output' in unique_columns_list):
            for instruction, output in zip(data['instruction'], data['output']):
                instruction = f"{instruction}"
                target_data = f"instruction : {instruction}\noutput : {output}"
                unlabeled_target_list.append(target_data)
            self.unlabeled_target_list = unlabeled_target_list[:17000]

        print(f"[Unlabeled Data Loaded!!! unlabel : {len(self.unlabeled_target_list)}]")

    def load_labeled_data(self):
        """
        라벨링이 되어있는 데이터를 로드 합니다.
        """

        if self.labeled_data_path.endswith('.csv'):
            data = pd.read_csv(self.labeled_data_path)
        
        unique_task_list = list(data['task'].unique())

        print(f"task : [{unique_task_list}]")
        
        unique_columns_list = list(data.columns.unique())
        
        labeled_target_dict = dict()
        labeled_target_list = list()

        # instruction, output, input, task가 column으로 주어질때
        if ('instruction' in unique_columns_list) and ('output' in unique_columns_list) and ('input' in unique_columns_list):
            for instruction, output, input, task in zip(data['instruction'], data['output'], data['input'], data['task']):
                instruction = f"{instruction}\n{input}"
                task = unique_task_list.index(task)
                target_data = f"instruction : {instruction}\noutput : {output}"
                labeled_target_dict = {
                    'target' : target_data,
                    'label' : task
                }
                labeled_target_list.append(labeled_target_dict)
            self.labeled_target_list = labeled_target_list


        # instruction, output, task가 column으로 주어질때
        elif ('instruction' in unique_columns_list) and ('output' in unique_columns_list):
            for instruction, output, task in zip(data['instruction'], data['output'], data['task']):
                instruction = f"{instruction}"
                task = unique_task_list.index(task)
                target_data = f"instruction : {instruction}\noutput : {output}"
                labeled_target_dict = {
                    'target' : target_data,
                    'label' : task
                }
                labeled_target_list.append(labeled_target_dict)
            self.labeled_target_list = labeled_target_list

        print(f"[Labeled Data Loaded!!! label : {len(self.labeled_target_list)}]")

    def concat_data(self):
        """
        라벨링이 되지 않은 데이터와 라벨링이 된 데이터를 합칩니다.
        -> 토크나이저를 직접 생성할때 사용하기 위해
        """

        self.all_target_list = self.unlabeled_target_list + [data['target'] for data in self.labeled_target_list]
    
    def prepare_tokenizer(self):
        """
        토크나이저를 준비합니다. 
        use_hf_tokenizer가 True라면 허깅페이스 토크나이저를 가져옵니다.
        False라면 입력된 모든 데이터를 기반으로 토크나이저를 생성합니다.
        이미 토크나이저가 생성되어 있다면 바로 load 합니다.
        """

        if self.use_hf_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_path, truncation = True)

        else :
            already_tokenizer_path = os.path.join(f'../IDoFew_data/tokenizer/new_resources', f"pecab_sp-{int(self.vocab_size)//1000}k/tok.model")
            if not os.path.exists(already_tokenizer_path):
                Generate_Tokenizer = make_tokenizer(self.tokenizer_path, self.all_target_list, self.vocab_size)
                Generate_Tokenizer.make_tokenizer_run()

            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(already_tokenizer_path)