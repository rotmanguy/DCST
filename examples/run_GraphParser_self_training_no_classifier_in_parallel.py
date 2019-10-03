import ray
import subprocess
import torch

dict_ray_args = {}
use_gpu = torch.cuda.is_available()
if use_gpu:
    dict_ray_args['num_gpus']=1
    ray.init(num_gpus=torch.cuda.device_count())
else:
    dict_ray_args['num_cpus'] = 1
    ray.init()

@ray.remote(**dict_ray_args)
def call_GraphParser(domain, dataset, num_of_train_samples):
    if dataset == 'ontonotes':
        shell_str = ['python examples/GraphParser_self_training_no_sequence_tagger.py',
                     '--dataset', dataset,
                     '--domain', domain,
                     '--rnn_mode LSTM --num_epochs 70 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding glove --word_path "data/glove.840B.300d.txt" --char_embedding random',
                     '--model_path', "saved_models/" + dataset + "_parser_sl_no_sequence_tagger_" + domain + "_" + str(num_of_train_samples) + "_new/",
                     '--parser_path "saved_models/' +dataset + '_parser_' + domain + '_' + str(num_of_train_samples) + '/"']
    elif dataset == 'ud':
        shell_str = ['python examples/GraphParser_self_training_no_sequence_tagger.py',
                     '--dataset', dataset,
                     '--domain', domain,
                     '--rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_path "data/multilingual_word_embeddings/cc."' + domain + '".300.vec"',
                     '--word_embedding fasttext',
                     '--char_embedding random',
                     '--model_path', "saved_models/" + dataset + "_parser_sl_no_sequence_tagger_" + domain + "_" + str(num_of_train_samples) + "_wpt/",
                     '--parser_path "saved_models/' +dataset + '_parser_' + domain + '_' + str(num_of_train_samples) + '_wpt/"']
    else:
        raise ValueError
    p = subprocess.Popen(' '.join(shell_str), shell=True)
    p.communicate()

datasets = ['ud']
# num_of_train_samples_list = [100, 1000]
num_of_train_samples_list = [100, 500, 1000]
calls_list = []
for dataset in datasets:
    if dataset == 'ontonotes':
        domains = ['all', 'bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
    elif dataset == 'ud':
        domains = ['cu', 'da', 'fa', 'id', 'lv', 'sl', 'sv', 'tr', 'ur', 'vi']
    else:
        raise ValueError
    for num_of_train_samples in num_of_train_samples_list:
        for domain in domains:
            calls_list.append(call_GraphParser.remote(domain, dataset, num_of_train_samples))
ray.get(calls_list)