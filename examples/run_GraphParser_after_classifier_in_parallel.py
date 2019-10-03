import ray
import subprocess
import torch

dict_ray_args = {}
use_gpu = torch.cuda.is_available()
if use_gpu:
    dict_ray_args['num_gpus']=1
    ray.init(num_gpus=torch.cuda.device_count(), temp_dir='/tmp/ray_6')
    #ray.init(num_gpus=torch.cuda.device_count())
else:
    dict_ray_args['num_cpus'] = 1
    ray.init()

@ray.remote(**dict_ray_args)
def call_GraphParser(domain, dataset, task, num_of_train_samples, use_unlabeled_data, use_labeled_data):
    if dataset == 'ontonotes':
        # # # Normal
        # shell_str = ['python examples/GraphParser.py',
        #              '--dataset', dataset,
        #              '--domain', domain,
        #              '--rnn_mode LSTM --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
        #              "--punct_set '.' '``' "''" ':' ',' ",
        #              '--word_embedding glove --word_path "data/glove.840B.300d.txt" --char_embedding random',
        #              '--num_epochs 100',
        #              '--set_num_training_samples ' + str(num_of_train_samples)
        #              #'--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + 'rel_sons_depth' + "_" + str(num_of_train_samples) + "_gating/",
        #              #'--model_path', "saved_models/" + dataset + "_parser_nw_" + domain + "_" + 'rel_sons_depth' + "_gating_da/",
        #              #'--load_path', "saved_models/" + dataset + "_parser_" + domain + "_" + str(num_of_train_samples) + "/domain_" + domain + ".pt",
        #              #'--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task[0] + "_" + str(num_of_train_samples) + "/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task[1] + "_" + str(num_of_train_samples) + "/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task[2] + "_" + str(num_of_train_samples) + "/domain_" + domain + ".pt"
        #              #'--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + 'nw_' + domain + "_" + task[0] + "_da" + "/src_domain_nw_tgt_domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + 'nw_' + domain + "_" + task[1] + "_da" + "/src_domain_nw_tgt_domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + 'nw_' + domain + "_" + task[2] + "_da" + "/src_domain_nw_tgt_domain_" + domain + ".pt"
        #              ]
        # if task == 'rel_sons_depth':
        #     shell_str += ['--gating --num_gates 4']
        # else:
        #     shell_str += ['--gating --num_gates 2']
        # if use_unlabeled_data and use_labeled_data:
        #     if task == 'rel_sons_depth':
        #         shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'relative_pos_based' + "_" + str(
        #             num_of_train_samples) + "/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'number_of_children' + "_" + str(
        #             num_of_train_samples) + "/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'distance_from_the_root' + "_" + str(
        #             num_of_train_samples) + "/domain_" + domain + ".pt"]
        #     else:
        #         shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(
        #             num_of_train_samples) + "/domain_" + domain + ".pt"]
        #     shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_gating/"]
        # elif use_unlabeled_data and not use_labeled_data:
        #     if task == 'rel_sons_depth':
        #         shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'relative_pos_based' + "_" + str(
        #             num_of_train_samples) + "_unlabeled/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'number_of_children' + "_" + str(
        #             num_of_train_samples) + "_unlabeled/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'distance_from_the_root' + "_" + str(
        #             num_of_train_samples) + "_unlabeled/domain_" + domain + ".pt"]
        #     else:
        #         shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(
        #             num_of_train_samples) + "_unlabeled/domain_" + domain + ".pt"]
        #     shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_gating_unlabeled/"]
        # elif not use_unlabeled_data and use_labeled_data:
        #     if task == 'rel_sons_depth':
        #         shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'relative_pos_based' + "_" + str(
        #             num_of_train_samples) + "_labeled/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'number_of_children' + "_" + str(
        #             num_of_train_samples) + "_labeled/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'distance_from_the_root' + "_" + str(
        #             num_of_train_samples) + "_labeled/domain_" + domain + ".pt"]
        #     else:
        #         shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(
        #             num_of_train_samples) + "_labeled/domain_" + domain + ".pt"]
        #     shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_gating_labeled/"]
        # else:
        #     raise ValueError
        # # For DA
        source_domain = 'nw'
        shell_str = ['python examples/GraphParser_for_da.py',
                     '--dataset', dataset,
                     '--src_domain', source_domain,
                     '--domain', domain,
                     '--rnn_mode LSTM --num_epochs 70 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding glove --word_path "data/glove.840B.300d.txt" --char_embedding random',
                     #'--model_path', "saved_models/" + dataset + "_parser_" + source_domain + "_" + domain + "_" + task + "_gating_da/",
                     '--model_path', "saved_models/" + dataset + "_parser_" + source_domain + "_" + domain + "_" + task + "_gating_da_freeze_cls/",
                     #'--load_path', "saved_models/ontonotes_parser_nw_full_train/domain_nw.pt",
                     '--freeze_sequence_taggers'
                     ]
        if task == 'rel_sons_depth':
            shell_str += ['--gating --num_gates 4']
            '--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + source_domain + "_" + domain + "_" + \
            'relative_pos_based' + "_da/src_domain_" + source_domain + "_tgt_domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + source_domain + "_" + domain + "_" + \
            'number_of_children' + "_da/src_domain_" + source_domain + "_tgt_domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + source_domain + "_" + domain + "_" + \
            'distance_from_the_root' + "_da/src_domain_" + source_domain + "_tgt_domain_" + domain + ".pt"
        else:
            shell_str += ['--gating --num_gates 2']
            shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + source_domain + "_" + domain + "_" + \
            task + "_da/src_domain_" + source_domain + "_tgt_domain_" + domain + ".pt"]
    elif dataset == 'ud':
        shell_str = ['python examples/GraphParser.py',
                     '--dataset', dataset,
                     '--domain', domain,
                     '--rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding fasttext',
                     '--word_path "data/multilingual_word_embeddings/cc."' + domain + '".300.vec"',
                     '--char_embedding random',
                     '--set_num_training_samples ' + str(num_of_train_samples),
                     #'--load_path', "saved_models/" + dataset + "_parser_" + domain + "_"+str(num_of_train_samples)+"_wpt/domain_" + domain + ".pt",
                     #'--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task[0] + "_" + str(num_of_train_samples) + "_wpt/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task[1] + "_" + str(num_of_train_samples) + "_wpt/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task[2] + "_" + str(num_of_train_samples) + "_wpt/domain_" + domain + ".pt"
                     ]
        if task == 'rel_sons_depth':
            shell_str += ['--gating --num_gates 4']
        else:
            shell_str += ['--gating --num_gates 2']
        if use_unlabeled_data and use_labeled_data:
            if task == 'rel_sons_depth':
                shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'relative_pos_based' + "_" + str(num_of_train_samples) + "_wpt/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'number_of_children' + "_" + str(num_of_train_samples) + "_wpt/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'distance_from_the_root' + "_" + str(num_of_train_samples) + "_wpt/domain_" + domain + ".pt"]
                shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_gating_wpt/"]
            else:
                shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_wpt/domain_" + domain + ".pt"]
                shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_gating_wpt/"]
        elif use_unlabeled_data and not use_labeled_data:
            if task == 'rel_sons_depth':
                shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'relative_pos_based' + "_" + str(num_of_train_samples) + "_wpt_unlabeled/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'number_of_children' + "_" + str(num_of_train_samples) + "_wpt_unlabeled/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'distance_from_the_root' + "_" + str(num_of_train_samples) + "_wpt_unlabeled/domain_" + domain + ".pt"]
                shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_gating_wpt_unlabeled/"]
            else:
                shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_wpt_unlabeled/domain_" + domain + ".pt"]
                shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_gating_wpt_unlabeled/"]
        elif not use_unlabeled_data and use_labeled_data:
            if task == 'rel_sons_depth':
                shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'relative_pos_based' + "_" + str(num_of_train_samples) + "_wpt_labeled/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'number_of_children' + "_" + str(num_of_train_samples) + "_wpt_labeled/domain_" + domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + 'distance_from_the_root' + "_" + str(num_of_train_samples) + "_wpt_labeled/domain_" + domain + ".pt"]
                shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_gating_wpt_labeled/"]
            else:
                shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_wpt_labeled/domain_" + domain + ".pt"]
                shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + domain + "_" + task + "_" + str(num_of_train_samples) + "_gating_wpt_labeled/"]
        else:
            raise ValueError
    else:
        raise ValueError
    command = ' '.join(shell_str)
    print(command)
    #p = subprocess.Popen(command, shell=True)
    #p.communicate()

dataset = 'ud'
if dataset == 'ontonotes':
    #domains = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
    ## for DA
    domains = ['bc', 'bn', 'mz', 'pt', 'tc', 'wb']
elif dataset == 'ud':
    domains = ['cu', 'da', 'fa', 'id', 'lv', 'sl', 'sv', 'tr', 'ur', 'vi']
    domains = ['cu']
else:
    raise ValueError

tasks = ['relative_pos_based']
calls_list = []
for task in tasks:
    for domain in domains:
        calls_list.append(call_GraphParser.remote(domain, dataset, task, 500, True, False))

# tasks = ['rel_sons_depth', 'relative_pos_based', 'number_of_children', "distance_from_the_root", "language_model"]
# tasks = ['rel_sons_depth']
# flags = [(True, False), (False, True)]
# num_of_train_samples_list = [500, 100, 1000]
# num_of_train_samples_list = [100]
# calls_list = []
# for num_of_train_samples in num_of_train_samples_list:
#     for domain in domains:
#         for (use_unlabeled_data, use_labeled_data) in flags:
#             for task in tasks:
#                 calls_list.append(call_GraphParser.remote(domain, dataset, task, num_of_train_samples, use_unlabeled_data, use_labeled_data))
ray.get(calls_list)
ray.shutdown()