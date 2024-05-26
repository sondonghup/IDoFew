import argparse
from data_loader.data_load import data_load_manager

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--unlabeled_data_path', required = True)
    parser.add_argument('--labeled_data_path', required = True)
    parser.add_argument('--n_clusters', type = int, required = True)
    parser.add_argument('--embed_model_path', required = True)
    parser.add_argument('--verbose', action = 'store_true')
    parser.add_argument('--tokenizer_path', required = True)
    parser.add_argument('--vocab_size')
    parser.add_argument('--use_hf_tokenizer', action = 'store_true')
    parser.add_argument('--hf_tokenizer_path')
    parser.add_argument('--pretrained_model_path', required = True)
    parser.add_argument('--batch_size', type = int, required = True, default = 64)
    parser.add_argument('--epoch', type = int, required = True, default = 10)
    parser.add_argument('--learning_rate', type = float, required = True, default = 3e-5)
    parser.add_argument('--save_path', type = str, required = True)
    args = parser.parse_args()

    IDoFew = data_load_manager(args.unlabeled_data_path,
                               args.labeled_data_path,
                               args.n_clusters,
                               args.embed_model_path,
                               args.verbose,
                               args.tokenizer_path,
                               args.vocab_size,
                               args.use_hf_tokenizer,
                               args.pretrained_model_path,
                               args.batch_size,
                               args.epoch,
                               args.learning_rate,
                               args.save_path,
                               args.hf_tokenizer_path)
    IDoFew.load_unlabeled_data()
    IDoFew.load_labeled_data()
    IDoFew.concat_data()
    IDoFew.prepare_tokenizer()
    IDoFew.ptm_sib_run()
    IDoFew.ptm_kmeans_run()
    IDoFew.ptm_ft_run()


