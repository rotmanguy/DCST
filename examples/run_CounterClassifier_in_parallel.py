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
def call_SequenceTagger(domain, dataset, task, num_of_train_samples, use_unlabeled_data, use_labeled_data):
    if dataset == 'ontonotes':
        shell_str = ['python examples/SequenceTagger.py',
                     '--dataset', dataset,
                     '--domain', domain,
                     '--task', task,
                     '--rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding glove --word_path "data/glove.840B.300d.txt" --char_embedding random',
                     '--parser_path', "saved_models/" + dataset + "_parser_" + domain + "_" + str(num_of_train_samples) + "/"]
        if use_unlabeled_data:
            shell_str += ['--use_unlabeled_data']
        if use_labeled_data:
            shell_str += ['--use_labeled_data']
        if use_unlabeled_data and use_labeled_data:
            shell_str += ['--model_path', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(
                num_of_train_samples) + "/"]
        elif use_unlabeled_data and not use_labeled_data:
            shell_str += ['--model_path', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(
                num_of_train_samples) + "_unlabeled/"]
        elif not use_unlabeled_data and use_labeled_data:
            shell_str += ['--model_path', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(
                num_of_train_samples) + "_labeled/"]
        else:
            raise ValueError

    elif dataset == 'ud':
        shell_str = ['python examples/SequenceTagger.py',
                     '--dataset', dataset,
                     '--domain', domain,
                     '--task', task,
                     '--rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding fasttext',
                     '--word_path "data/multilingual_word_embeddings/cc."' + domain + '".300.vec"',
                     '--char_embedding random',
                     '--parser_path', "saved_models/" + dataset + "_parser_" + domain + "_" + str(num_of_train_samples) + "_wpt/"]
        if use_unlabeled_data:
            shell_str += ['--use_unlabeled_data']
        if use_labeled_data:
            shell_str += ['--use_labeled_data']
        if use_unlabeled_data and use_labeled_data:
            shell_str += ['--model_path', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(
                num_of_train_samples) + "_wpt/"]
        elif use_unlabeled_data and not use_labeled_data:
            shell_str += ['--model_path', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(
                num_of_train_samples) + "_wpt_unlabeled/"]
        elif not use_unlabeled_data and use_labeled_data:
            shell_str += ['--model_path', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(
                num_of_train_samples) + "_wpt_labeled/"]
        else:
            raise ValueError
    else:
        raise ValueError
    command = ' '.join(shell_str)
    print(command)
    #p = subprocess.Popen(command, shell=True)
    #p.communicate()

calls_list = []
datasets = ['ud']
flags = [(True, False), (False, True)]
for dataset in datasets:
    if dataset == 'ontonotes':
        domains = ['all', 'bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
        domains = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']

    elif dataset == 'ud':
        domains = ['cu', 'da', 'fa', 'id', 'lv', 'sl', 'sv', 'tr', 'ur', 'vi']
        doamins = ['cu']
    else:
        raise ValueError
    tasks = ['relative_pos_based', 'number_of_children', 'distance_from_the_root', 'language_model']
    tasks = ['relative_pos_based', 'number_of_children', 'distance_from_the_root']
    num_of_train_samples_list = [500, 100, 1000]
    num_of_train_samples_list = [100]
    for num_of_train_samples in num_of_train_samples_list:
        for task in tasks:
            for domain in domains:
                for (use_unlabeled_data, use_labeled_data) in flags:
                    calls_list.append(call_SequenceTagger.remote(domain, dataset, task, num_of_train_samples, use_unlabeled_data, use_labeled_data))
ray.get(calls_list)