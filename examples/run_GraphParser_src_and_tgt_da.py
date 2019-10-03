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
def call_GraphParser(src_domain, tgt_domain, dataset):
    if dataset == 'ontonotes':
        shell_str = ['python examples/GraphParser_src_and_tgt_da.py',
                     '--dataset', dataset,
                     '--src_domain', src_domain,
                     '--tgt_domain', tgt_domain,
                     '--rnn_mode LSTM --num_epochs 70 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5',
                     "--punct_set '.' '``' "''" ':' ',' ",
                     '--word_embedding glove --word_path "data/glove.840B.300d.txt" --char_embedding random'
                     ]
    else:
        raise ValueError

    shell_str += ['--model_path', "saved_models/" + dataset + "_parser_" + src_domain + "_" + tgt_domain + '_src_and_tgt_joint_training_da']
    p = subprocess.Popen(' '.join(shell_str), shell=True)
    p.communicate()

dataset = 'ontonotes'
if dataset == 'ontonotes':
    src_domains = ['nw']
    tgt_domains = ['wb', 'bc', 'bn', 'mz', 'pt', 'tc']
else:
    raise ValueError


calls_list = []
for src_domain in src_domains:
    for tgt_domain in tgt_domains:
        calls_list.append(call_GraphParser.remote(src_domain, tgt_domain, dataset))

ray.get(calls_list)