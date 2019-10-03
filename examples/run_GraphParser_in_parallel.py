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
def call_GraphParser(domain, dataset, gating, num_of_train_samples):
    if dataset == 'ontonotes':
        # shell_str = ['python examples/GraphParser.py',
        #              '--dataset', dataset,
        #              '--domain', domain,
        #              '--rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
        #              "--punct_set '.' '``' "''" ':' ',' ",
        #              '--word_embedding glove --word_path "data/glove.840B.300d.txt" --char_embedding random --set_num_training_samples ' + str(num_of_train_samples)
        #              ]
        # Full training
        shell_str = ['python examples/GraphParser.py',
                     '--dataset', dataset,
                     '--domain', domain,
                     '--rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding glove --word_path "data/glove.840B.300d.txt" --char_embedding random',
                     ]
    elif dataset == 'ud':
        # shell_str = ['python examples/GraphParser.py',
        #              '--dataset', dataset,
        #              '--domain', domain,
        #              '--rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
        #              "--punct_set '.' '``' "''" ':' ',' ",
        #              '--word_embedding fasttext',
        #              '--word_path "data/multilingual_word_embeddings/cc."' + domain + '".300.vec"',
        #              '--char_embedding random --set_num_training_samples ' + str(num_of_train_samples)
        #              ]

    # Full training
        shell_str = ['python examples/GraphParser.py',
                     '--dataset', dataset,
                     '--domain', domain,
                     '--rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc."' + domain.split('_')[0] + '".300.vec" --char_embedding random'
                     ]
    else:
        raise ValueError
    if gating:
        shell_str += ["--gating --num_gates 4"]
        if dataset == 'ontonotes':
            if num_of_train_samples is not None:
                shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + str(
                num_of_train_samples) + "_gating_4_gates/"]
            else:
                shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_gating_4_gates/"]
        elif dataset == 'ud':
            if num_of_train_samples is not None:
                shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + str(num_of_train_samples) + "_wpt_gating_4_gates/"]
            else:
                shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_wpt_gating_4_gates/"]

        else:
            raise ValueError
    else:
        # with set_num_train_samples
        #shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + str(num_of_train_samples) + '_wpt/']
        # DA
        # shell_str += ['--model_path', "saved_models/" + dataset + "_parser_nw_" + domain + "_da"]
        #full train
        if dataset == 'ontonotes':
            shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_full_train"]
        elif dataset == 'ud':
            shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_full_train_wpt"]
        else:
            raise ValueError
    command = ' '.join(shell_str)
    print(command)
    #p = subprocess.Popen(command, shell=True)
    #p.communicate()

calls_list = []
datasets = ['ud']
gatings = [True]
for dataset in datasets:
    for gating in gatings:
        if dataset == 'ontonotes':
            domains = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
        elif dataset == 'ud':
            domains = ['cu']
        else:
            raise ValueError
        for domain in domains:
            calls_list.append(call_GraphParser.remote(domain, dataset, gating, None))

#num_of_train_samples_list = [100, 500, 1000]

# NLP 11 run
# gatings = [False, True]
# num_of_train_samples_list = [100, 500, 1000]
# calls_list = []
# for gating in gatings:
#     for domain in domains:
#         for num_of_train_samples in num_of_train_samples_list:
#             if domain == 'sv' and num_of_train_samples == 100:
#                 continue
#             if domain == 'tr' and num_of_train_samples == 100:
#                 continue
#             calls_list.append(call_GraphParser.remote(domain, dataset, gating, num_of_train_samples))



ray.get(calls_list)