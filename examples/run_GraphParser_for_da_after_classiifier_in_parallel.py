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
def call_GraphParser(src_domain, tgt_domain, dataset, task, freeze):
    if dataset == 'ontonotes':
        # # For DA
        shell_str = ['python examples/GraphParser_for_da.py',
                     '--dataset', dataset,
                     '--src_domain', src_domain,
                     '--tgt_domain', tgt_domain,
                     '--rnn_mode LSTM --num_epochs 70 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding glove --word_path "data/glove.840B.300d.txt" --char_embedding random',
                     #'--model_path', "saved_models/" + dataset + "_parser_" + source_domain + "_" + domain + "_" + task + "_gating_da/",
                     #'--load_path', "saved_models/ontonotes_parser_nw_full_train/domain_nw.pt",
                     ]
        if task == 'rel_sons_depth':
            shell_str += ['--gating --num_gates 4']
            '--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + \
            'rel_pos_based' + "_da/src_domain_" + src_domain + "_tgt_domain_" + tgt_domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + \
            'num_of_sons' + "_da/src_domain_" + src_domain + "_tgt_domain_" + tgt_domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + \
            'tree_depth' + "_da/src_domain_" + src_domain + "_tgt_domain_" + tgt_domain + ".pt"
        else:
            shell_str += ['--gating --num_gates 2']
            shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + \
                          task + "_da/src_domain_" + src_domain + "_tgt_domain_" + tgt_domain + ".pt"]
        if freeze:
            shell_str += ['--freeze_sequence_taggers']
            shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + src_domain + "_" + tgt_domain + "_" + task + "_gating_da_freeze_cls/"]
        else:
            shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + src_domain + "_" + tgt_domain + "_" + task + "_gating_da/"]

    elif dataset == 'ud':
        shell_str = ['python examples/GraphParser_for_da.py',
                     '--dataset', dataset,
                     '--src_domain', src_domain,
                     '--tgt_domain', tgt_domain,
                     '--rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding fasttext',
                     '--word_path "data/multilingual_word_embeddings/cc.' + tgt_domain.split('_')[0] + '.300.vec"',
                     '--char_embedding random']
        if task == 'rel_sons_depth':
            shell_str += ['--gating --num_gates 4']
        else:
            shell_str += ['--gating --num_gates 2']
        if task == 'rel_sons_depth':
            shell_str += ['--gating --num_gates 4']
            '--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + \
            'rel_pos_based' + "_wpt_da/src_domain_" + src_domain + "_tgt_domain_" + tgt_domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + \
            'num_of_sons' + "_wpt_da/src_domain_" + src_domain + "_tgt_domain_" + tgt_domain + ".pt" + " saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + \
            'tree_depth' + "_wpt_da/src_domain_" + src_domain + "_tgt_domain_" + tgt_domain + ".pt"
        else:
            shell_str += ['--load_sequence_taggers_paths', "saved_models/" + dataset + "_sequence_tagger_" + src_domain + "_" + tgt_domain + "_" + \
                          task + "_wpt_da/src_domain_" + src_domain + "_tgt_domain_" + tgt_domain + ".pt"]
        if freeze:
            shell_str += ['--freeze_sequence_taggers']
            shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + src_domain + "_" + tgt_domain + "_" + task + "_gating_wpt_da_freeze_cls/"]
        else:
            shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + src_domain + "_" + tgt_domain + "_" + task + "_gating_wpt_da/"]

    else:
        raise ValueError
    command = ' '.join(shell_str)
    print(command)
    # p = subprocess.Popen(command, shell=True)
    # p.communicate()

dataset = 'ud'
if dataset == 'ontonotes':
    ## for DA
    src_domains = ['nw']
    tgt_domains = ['bc', 'bn', 'mz', 'pt', 'tc', 'wb']
elif dataset == 'ud':
    #src_domains = ['cs_fictree', 'cs_pdt', 'gl_ctg', 'gl_treegal', 'it_isdt', 'it_partut', 'it_postwita', 'ro_nonstandard', 'ro_rrt']
    #tgt_domains = ['cs_fictree', 'cs_pdt', 'gl_ctg', 'gl_treegal', 'it_isdt', 'it_partut', 'it_postwita', 'ro_nonstandard', 'ro_rrt']
    src_domains = ['gl_ctg']
    tgt_domains = ['gl_treegal']
else:
    raise ValueError

freezes = [True, False]
tasks = ['relative_pos_based']
calls_list = []
for freeze in freezes:
    for task in tasks:
        for src_domain in src_domains:
            for tgt_domain in tgt_domains:
                if src_domain.split('_')[0] != tgt_domain.split('_')[0] or src_domain == tgt_domain:
                    continue
                calls_list.append(call_GraphParser.remote(src_domain, tgt_domain, dataset, task, freeze))

# tasks = ['rel_sons_depth', 'rel_pos_based', 'num_of_sons', "tree_depth", "lm"]
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