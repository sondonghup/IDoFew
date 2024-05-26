python3 -m run \
    --unlabeled_data_path  \
    --labeled_data_path  \
    --n_clusters \
    --embed_model_path  \
    --verbose \
    --tokenizer_path  \
    --use_hf_tokenizer \
    --hf_tokenizer_path  \
    --vocab_size  \
    --pretrained_model_path  \
    --batch_size  \
    --epoch  \
    --learning_rate  \
    --save_path model/

    # parser.add_argument('--unlabeled_data_path', required = True)
    # parser.add_argument('--labeled_data_path', required = True)
    # parser.add_argument('--n_clusters', type = int, required = True)
    # parser.add_argument('--embed_model_path', required = True)
    # parser.add_argument('--verbose', action = 'store_true')
    # parser.add_argument('--tokenizer_path', required = True)
    # parser.add_argument('--vocab_size')
    # parser.add_argument('--use_hf_tokenizer', action = 'store_true')
    # parser.add_argument('--hf_tokenizer_path')
    # parser.add_argument('--pretrained_model_path', required = True)
    # parser.add_argument('--batch_size', type = int, required = True, default = 64)
    # parser.add_argument('--epoch', type = int, required = True, default = 10)
    # parser.add_argument('--learning_rate', type = float, required = True, default = 3e-5)
    # parser.add_argument('--save_path', type = str, required = True)