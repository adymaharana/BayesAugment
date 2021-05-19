#python postprocessing.py --output_file out/squad_ques_dev.out --corenlp_cache_file ../AutoAdverse/data/dev_split_v2_corenlp_cache.json --dataset_file ../AutoAdverse/data/dev-split-v2.0.json --glove_file ../data/glove.840B.300d.txt

#python postprocessing.py --output_file out/squad_ques_train.out --corenlp_cache_file ../AutoAdverse/data/train_v2_corenlp_cache.json --dataset_file ../AutoAdverse/data/train-v2.0.json --glove_file ../data/glove.840B.300d.txt

#python postprocessing.py --output_file out/newsqa_ques_train.out --corenlp_cache_file ../data/newsqa/train_corenlp_cache.json --dataset_file ../data/newsqa/train.json --glove_file ../data/glove.840B.300d.txt --dataset_type newsqa

python postprocessing.py --output_file out/newsqa_ques_dev.out --corenlp_cache_file ../data/newsqa/dev_corenlp_cache.json --dataset_file ../data/newsqa/dev.json --glove_file ../data/glove.840B.300d.txt --dataset_type newsqa