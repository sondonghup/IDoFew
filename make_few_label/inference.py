from vllm import LLM, SamplingParams
import argparse
import pandas as pd
from tqdm import tqdm

class prompt_vllm():
    def __init__(self, model_path, prompt, stop_word):
        
        self.token = ''
        self.llm = LLM(model_path, tensor_parallel_size = 1)
        self.sampling_params = SamplingParams(temperature=0.1, top_p=0.01, max_tokens=4096, repetition_penalty=1.0, stop=stop_word, include_stop_str_in_output=True)
        self.prompt = prompt
        print('[Loaded!]')

    def data_load(self, data_path):

        def make_prompt(text):
            prompt = f'''SYSTEM : {{ {self.prompt} }}
                    USER : {{ {text} }}
                    출력 : 
                    '''
            return prompt

        self.target_datas = list()
        target_inputs = list()

        datas = pd.read_csv(data_path)
        for instruction, output in zip(datas['instruction'], datas['output']):
            instruction = instruction.replace('=======Request End=======', '')
            output = str(output).replace('=======Request End=======', '')
            target_input = 'question : ' + instruction
            target_inputs.append([instruction, output])
            self.target_datas.append(make_prompt(target_input))

        print('[Data Loaded!]')

        return target_inputs
            

    def generate(self):
        output = self.llm.generate(self.target_datas, self.sampling_params)

        return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default = '' )
    parser.add_argument('--stop_word', default = '[종료]')
    parser.add_argument('--target_data', default = '' )
    parser.add_argument('--prompt_path', default = 'prompt_sum.txt')
    parser.add_argument('--output_path', default = '')
    args = parser.parse_args()

    with open(args.prompt_path, 'r', encoding='utf-8')as f:
        prompt = ''.join([text for text in f])
    
    LLM = prompt_vllm(args.model_path, prompt, args.stop_word)
    target_inputs = LLM.data_load(args.target_data)
    outputs = LLM.generate()

    converted_outputs = list()
    converted_dict = dict()

    for output, target_input in tqdm(zip(outputs, target_inputs)):
        converted_dict = {
            'instruction' : target_input[0],
            'output' : target_input[1],
            'score' : output.outputs[0].text
        }
        converted_outputs.append(converted_dict)
        
    df = pd.DataFrame(converted_outputs)
    df.to_csv(args.output_path, index=False)