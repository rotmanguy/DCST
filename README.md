Official code for the paper ["Deep Contextualized Self-training for Low Resource Dependency Parsing"](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00294).\
If you use this code please cite our paper.

# Requirements
Simply run:
```
pip install -r requirements.txt
```
# Data
First download the Universal Dependencies treebanks from https://universaldependencies.org/ and unzip the folder.
Then, run the script: \
`python utils/io_/convert_ud_to_onto_format.py --ud_data_path "path"` \
(e.g. `python utils/io_/convert_ud_to_onto_format.py --ud_data_path /Data/Datasets/Universal_Dependencies_2.3/ud-treebanks-v2.3`)

## Multilingual Word Embeddings
It is also required to download the FastText multilingual word embeddings for the UD languages.
They can be downloaded from here: \
https://fasttext.cc/docs/en/crawl-vectors.html . \
Word embedding for cu (Old Church Slavonic) can be found in this link: \
https://github.com/mwydmuch/extremeText/blob/master/pretrained-vectors.md . \
Once unzipped, the multilingual word embedding (.vec extensions) should be placed under the `data/multilingual_word_embeddings` folder.

It is also optional to download the GloVe embeddings for English from: \
http://nlp.stanford.edu/data/glove.6B.zip .\
Once unzziped, the text file (.txt extension) should be placed under `data` folder.

# Low Resource In-domain Experiments
In order to run the low resource in-domain experiments there are three steps we need to follow:
1. Running the base Biaffine parser
2. Running the sequence tagger(s)
3. Running the combined DCST parser

## Running the base Biaffine Parser
Here is an example for training the entire da (Danish) data set:
```
python examples/GraphParser.py --dataset ud --domain da --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.da.300.vec" --char_embedding random --model_path saved_models/ud_parser_da_full_train
```

If you wish to to train only a subset of the training set, for example selecting 500 samples randomly, you should run:
```
python examples/GraphParser.py --dataset ud --domain da --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.da.300.vec" --char_embedding random --model_path saved_models/ud_parser_da_500 --set_num_training_samples 500
```
The remaining unlabeled data is labeled automatically at the end of the training.
## Running the Sequence Tagger
Once training the base parser, we can now run the Sequnece Tagger on any of the three proposed sequence tagging tasks in order to learn the syntactical contextualized word embeddings from the unlabeled data set.

Running the Relative Pos-based task:
```
python examples/SequenceTagger.py --dataset ud --domain da --task relative_pos_based --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.da.300.vec" --char_embedding random --parser_path saved_models/ud_parser_da_500/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_da_relative_pos_based_500_unlabeled/
```
Running the Number of Children task:
```
python examples/SequenceTagger.py --dataset ud --domain da --task number_of_children --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.da.300.vec" --char_embedding random --parser_path saved_models/ud_parser_da_500/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_da_number_of_children_500_unlabeled/
```
Running the Distance from the Root task:
```
python examples/SequenceTagger.py --dataset ud --domain da --task distance_from_the_root --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.da.300.vec" --char_embedding random --parser_path saved_models/ud_parser_da_500/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_da_distance_from_the_root_500_unlabeled/
```
We also allow the option of learning contextualized word embeddings from a Language Model on the unlabeled data:
```
python examples/SequenceTagger.py --dataset ud --domain da --task language_model --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.da.300.vec" --char_embedding random --parser_path saved_models/ud_parser_da_500/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_da_language_model_500_unlabeled/
```
## Final step - Running the Combined DCST Parser
As a final step we can now run the DCST (ensemble) parser:
```
python examples/GraphParser.py --dataset ud --domain da --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.da.300.vec" --char_embedding random --set_num_training_samples 500 --gating --num_gates 4 --load_sequence_taggers_paths saved_models/ud_sequence_tagger_da_relative_pos_based_500_unlabeled/domain_da.pt saved_models/ud_sequence_tagger_da_number_of_children_500_unlabeled/domain_da.pt saved_models/ud_sequence_tagger_da_distance_from_the_root_500_unlabeled/domain_da.pt --model_path saved_models/ud_parser_da_ensemble_500_gating_unlabeled/
```
If you wish to integrate the base parser with only one (or two) sequence taggers, simply change the `num_gates`, `load_sequence_taggers_paths` and `model_path` inputs accordingly.

# Cross-domain Experiments
In order to run the cross-domain experiments there are four steps we need to follow:
1. Running the base Biaffine parser (on the source domain)
2. Labeling the unlabeled target data
3. Running the sequence tagger(s) (on the target domain)
4. Running the combined DCST parser (on the source domain)

## Running the base Biaffine Parser
First, run the base parser on the source domain, for example:
```
python examples/GraphParser.py --dataset ud --domain cs_fictree --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.cs.300.vec" --char_embedding random --model_path saved_models/ud_parser_cs_fictree_full_train
```
## Labeling the Unlabeled Target Data
Next, we need to label the unlabeled target domain data by the trained parser:
```
python examples/GraphParser.py --dataset ud --domain cs_pdt --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.cs.300.vec" --char_embedding random --model_path saved_models/ud_parser_cs_fictree_full_train --eval_mode --strict --load_path saved_models/ud_parser_cs_fictree_full_train/domain_cs_fictree.pt
```
## Running the Sequence Tagger
Now, we can train the sequence tagger on the auto-labeled target data according to the different sequence tagging tasks.

Running the Relative Pos-based task:
```
python examples/SequenceTagger_for_DA.py --dataset ud --src_domain cs_fictree --tgt_domain cs_pdt --task relative_pos_based --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.cs.300.vec" --char_embedding random --parser_path saved_models/ud_parser_cs_fictree_full_train/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_cs_fictree_cs_pdt_relative_pos_based_unlabeled/
```
Running the Number of Children task:
```
python examples/SequenceTagger_for_DA.py --dataset ud --src_domain cs_fictree --tgt_domain cs_pdt --task number_of_children --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.cs.300.vec" --char_embedding random --parser_path saved_models/ud_parser_cs_fictree_full_train/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_cs_fictree_cs_pdt_number_of_children_unlabeled/
```
Running the Distance from the Root task:
```
python examples/SequenceTagger_for_DA.py --dataset ud --src_domain cs_fictree --tgt_domain cs_pdt --task distance_from_the_root --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.cs.300.vec" --char_embedding random --parser_path saved_models/ud_parser_cs_fictree_full_train/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_cs_fictree_cs_pdt_distance_from_the_root_unlabeled/
```
Running the Language Model task:
```
python examples/SequenceTagger_for_DA.py --dataset ud --src_domain cs_fictree --tgt_domain cs_pdt --task language_model --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --tag_space 128 --num_layers 3 --num_filters 100 --use_char  --use_pos --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.cs.300.vec" --char_embedding random --parser_path saved_models/ud_parser_cs_fictree_full_train/ --use_unlabeled_data --model_path saved_models/ud_sequence_tagger_cs_fictree_cs_pdt_language_model_unlabeled/
```
## Final step - Running the Combined DCST Parser
Finally, we can run the DCST (ensemble) parser in order to re-train the source domain with our contextualized word embeddings:
```
python examples/GraphParser_for_DA.py --dataset ud --src_domain cs_fictree --tgt_domain cs_pdt --rnn_mode LSTM --num_epochs 100 --batch_size 16 --hidden_size 512 --arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char --use_pos  --word_dim 300 --char_dim 100 --pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --unk_replace 0.5 --punct_set '.' '``'  ':' ','  --word_embedding fasttext --word_path "data/multilingual_word_embeddings/cc.cs.300.vec" --char_embedding random --gating --num_gates 4 --load_sequence_taggers_paths saved_models/ud_sequence_tagger_cs_fictree_cs_pdt_relative_pos_based_unlabeled/src_domain_cs_fictree_tgt_domain_cs_pdt.pt saved_models/ud_sequence_tagger_cs_fictree_cs_pdt_number_of_children_unlabeled/src_domain_cs_fictree_tgt_domain_cs_pdt.pt saved_models/ud_sequence_tagger_cs_fictree_cs_pdt_distance_from_the_root_unlabeled/src_domain_cs_fictree_tgt_domain_cs_pdt.pt --model_path saved_models/ud_parser_cs_fictree_cs_pdt_ensemble_gating/
```
If you wish to integrate the base parser with only one (or two) sequence taggers, simply change the `num_gates`, `load_sequence_taggers_paths` and `model_path` inputs accordingly.

There is also an option to freeze the sequence tagger encoders, simply by adding `--freeze` to the command line.

## Citation
```
@article{doi:10.1162/tacl\_a\_00294,
author = {Rotman, Guy and Reichart, Roi},
title = {Deep Contextualized Self-training for Low Resource Dependency Parsing},
journal = {Transactions of the Association for Computational Linguistics},
volume = {7},
number = {},
pages = {695-713},
year = {2019},
doi = {10.1162/tacl\_a\_00294},

URL = { 
        https://doi.org/10.1162/tacl_a_00294
    
},
eprint = { 
        https://doi.org/10.1162/tacl_a_00294
    
}
}
```