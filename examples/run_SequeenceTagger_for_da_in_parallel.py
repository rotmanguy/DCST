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
def call_SequenceTagger(src_domain, tgt_domain, dataset, task, use_unlabeled_data, use_labeled_data):
    if dataset == 'ontonotes':
        shell_str = ['python examples/SequenceTagger_for_da.py',
                     '--dataset', dataset,
                     '--src_domain', src_domain,
                     '--tgt_domain', tgt_domain,
                     '--task', task,
                     '--rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding glove --word_path "data/glove.840B.300d.txt" --char_embedding random',
                     #'--parser_path', "saved_models/" + dataset + "_parser_" + src_domain + '/']
                     '--parser_path', "saved_models/" + dataset + "_parser_" + src_domain + "_full_train/"]
        if use_unlabeled_data:
            shell_str += ['--use_unlabeled_data']
        if use_labeled_data:
            shell_str += ['--use_labeled_data']
        if use_unlabeled_data and use_labeled_data:
            shell_str += ['--model_path', "saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + task + "_da_unlabeled_and_labeled/"]
        elif use_unlabeled_data and not use_labeled_data:
            shell_str += ['--model_path', "saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + task + "_da/"]
        else:
            raise ValueError
    elif dataset == 'ud':
        shell_str = ['python examples/SequenceTagger_for_da.py',
                     '--dataset', dataset,
                     '--src_domain', src_domain,
                     '--tgt_domain', tgt_domain,
                     '--task', task,
                     '--rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding random --char_embedding random',
                     '--parser_path', "saved_models/" + dataset + "_parser_" + src_domain + "_da/"]
        if use_unlabeled_data:
            shell_str += ['--use_unlabeled_data']
        if use_labeled_data:
            shell_str += ['--use_labeled_data']
        if use_unlabeled_data and use_labeled_data:
            shell_str += ['--model_path', "saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + task + "_da_unlabeled_and_labeled/"]
        elif use_unlabeled_data and not use_labeled_data:
            shell_str += ['--model_path', "saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + task + "_da/"]
        else:
            raise ValueError
    else:
        raise ValueError
    command = ' '.join(shell_str)
    print(command)
    #p = subprocess.Popen(command, shell=True)
    #p.communicate()

calls_list = []
flags = [(True, True)]
dataset = 'ud'
if dataset == 'ontonotes':
    source_domains = ['nw']
    target_domains = ['bc', 'bn', 'mz', 'pt', 'tc', 'wb']
    # source_domains = ['wb_len_under_10']
    # target_domains = ['wb_len_over_10']
elif dataset == 'ud':
    source_domains = ['gl_ctg']
    target_domains = ['gl_treegal']
else:
    raise ValueError
tasks = ['relative_pos_based', 'number_of_children', 'distance_from_the_root', 'language_model']
for task in tasks:
    for src_domain in source_domains:
        for tgt_domain in target_domains:
            for (use_unlabeled_data, use_labeled_data) in flags:
                calls_list.append(call_SequenceTagger.remote(src_domain, tgt_domain, dataset, task, use_unlabeled_data, use_labeled_data))
ray.get(calls_list)