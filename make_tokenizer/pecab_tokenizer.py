from tqdm import tqdm
import os
from pecab import PeCab
import sentencepiece as spm

class make_tokenizer():
    """
    토크나이저를 생성합니다.
    """

    def __init__(self,
                 pecab_tokenized_dir,
                 data,
                 vocab_size
                 ):

        self.pecab = PeCab()
        self.vocab_size = vocab_size
        self.character_coverage = 1.0
        self.normalization_rule_name = 'identity'
        self.pad_piece = '[PAD]'
        self.unk_piece = '[UNK]'
        self.bos_piece = '[BOS]'
        self.eos_piece = '[EOS]'
        self.unk_surface = '[UNK]'
        self.special_symbols = '[CLS],[SEP],[MASK]'
        self.pecab_tokenized_dir = pecab_tokenized_dir
        self.data = data
        self.tokenizer_path = '../IDoFew_data/tokenizer/'
        self.output_dir = os.path.join(f'{self.tokenizer_path}new_resources', f"pecab_sp-{int(self.vocab_size)//1000}k/")

    def tokenize(self, text, space_symbol='▃'):

        text = text.strip()
        text_ptr = 0
        tokenized = []

        for mor in self.pecab.morphs(text):
            token = mor
            if text[text_ptr] == ' ':
                while text[text_ptr] == ' ':
                    text_ptr += 1
                assert text[text_ptr] == token[0]
                tokenized.append(space_symbol)

            tokenized.append(token)
            text_ptr += len(token)
        
        if not os.path.exists(self.tokenizer_path):
            os.makedirs(self.tokenizer_path)

        with open(self.tokenizer_path + self.pecab_tokenized_dir, 'a', encoding='utf-8') as f:
            f.write(' '.join([token for token in tokenized]) + '\n')

    def make_commands(self):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        cmd = f"--input={self.tokenizer_path + self.pecab_tokenized_dir} "
        cmd += f"--model_prefix={os.path.join(self.output_dir, 'tok')} "
        cmd += f"--vocab_size={self.vocab_size} "
        cmd += f"--model_type=bpe "
        cmd += f"--character_coverage={self.character_coverage} "
        cmd += f"--normalization_rule_name={self.normalization_rule_name} "
        cmd += f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
        cmd += f"--pad_piece={self.pad_piece} "
        cmd += f"--unk_piece={self.unk_piece} "
        cmd += f"--bos_piece={self.bos_piece} "
        cmd += f"--eos_piece={self.eos_piece} "
        cmd += f"--unk_surface={self.unk_surface} "
        cmd += f"--user_defined_symbols={self.special_symbols} "

        self.cmd = cmd

    def train_sentence_piece(self):
        
        spm.SentencePieceTrainer.Train(self.cmd)

    def text_cleaning(self, text):

        text = text.replace('\n', ' ')
        text = text.replace('instruction : ', '')
        text = text.replace('output : ', '')

        return text
    
    def make_tokenizer_run(self):

        print(self.tokenizer_path + self.pecab_tokenized_dir)
        
        for text in tqdm(self.data, desc = 'tokenizing'):
            
            text = self.text_cleaning(text)

            try:
                self.tokenize(text)
            except Exception as e:
                print(e)
                continue

        self.make_commands()
        self.train_sentence_piece()