from torch.utils.data import Dataset
import torch

class make_dataset(Dataset):
    """
    학습을 위한 데이터셋을 만듭니다.
    """

    def __init__(self, data, tokenizer, model_max_length):
        """
        data : 학습 데이터
        tokenizer : 토크나이저
        model_max_length : sentencepiece로 토크나이저를 생성하게 되면 truncation이 안되기 때문에 모델크기를 받아옵니다.
        """

        self._data = [text for text in data if len(text['target']) < model_max_length - 2] # 모델크기 보다 작은 데이터만 가져옵니다.
        self._tokenizer = tokenizer
        self._model_max_length = model_max_length
        if self._tokenizer.__class__.__name__ == 'BertTokenizerFast':
            self.pad_token_id = self._tokenizer.pad_token_id
        else:
            self.pad_token_id = self._tokenizer.pad_id()

    def __len__(self):

        return len(self._data)
    
    def __getitem__(self, index):

        row = self._data[index]
        
        if len(row.keys()) == 1:
            input_text = row['target']

            if self._tokenizer.__class__.__name__ == 'BertTokenizerFast':
                return {
                    'input_ids' : self._tokenizer(input_text)['input_ids'],
                }
            else:
                return {
                    'input_ids' : self._tokenizer.encode_as_ids(input_text),
                }

        else: 
            input_text = row['target']
            label = row['label']
            
            if self._tokenizer.__class__.__name__ == 'BertTokenizerFast':
                return {
                    'input_ids' : self._tokenizer(input_text)['input_ids'],
                    'labels' : label
                }
            else:
                return {
                    'input_ids' : self._tokenizer.encode_as_ids(input_text),
                    'labels' : label
                }
    
    def _padding(self, sequence, value, max_len):

        padded_data = sequence + [value] * (max_len - len(sequence))

        return padded_data
    
    def _attention_mask(self, sequence, max_len):
        
        attention_mask = [1] * len(sequence) + [0] * (max_len - len(sequence))

        return attention_mask
    
    def collate_fn(self, batch):

        max_len = max(len(row['input_ids']) for row in batch)

        if len(batch[0].keys()) == 1:
            input_ids = [self._padding(row['input_ids'], self.pad_token_id, max_len) for row in batch]
            attention_masks = [self._attention_mask(row['input_ids'], max_len) for row in batch]
    
            return {
                'input_ids' : torch.tensor(input_ids),
                'attention_masks' : torch.tensor(attention_masks),
            }

        else:
            input_ids = [self._padding(row['input_ids'], self.pad_token_id, max_len) for row in batch]
            attention_masks = [self._attention_mask(row['input_ids'], max_len) for row in batch]
            labels = [row['labels'] for row in batch]
    
            return {
                'input_ids' : torch.tensor(input_ids),
                'attention_masks' : torch.tensor(attention_masks),
                'labels' : torch.tensor(labels)
            }