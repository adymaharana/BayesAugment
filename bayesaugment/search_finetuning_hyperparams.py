from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
import logging, argparse
import numpy as np
import glob
import random, json

from transformers import (WEIGHTS_NAME, BertConfig,
                                  XLMConfig, XLNetConfig, RobertaConfig,
                                  RobertaTokenizer)
from modelling_roberta import RobertaForQuestionAnswering
import torch
import os

# from adversarial_utils_squad import augment_with_adversaries as augment_squad_with_adversaries
#from adversarial_utils_newsqa import augment_with_adversaries as augment_newsqa_with_adversaries

from squad_roberta import load_and_cache_examples
from finetuned_roberta import trainRobertaForQA, initializeRobertaForQA, evaluateRobertaForQA

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer)
}

from argparse import Namespace

base_args = Namespace(**{'local_rank': -1,
                         'train_file': '../AutoAdverse/data/train-AutoAugment-QANet-SQuAD.json',
                         'predict_file': '../AutoAdverse/data/dev-split-v2.0.json',
                         'output_dir': './out/squad-bayes-AutoAugment/',
#                         'train_file': '../AutoAdverse/data/train-BayesAugment-S2G.json',
#                         'predict_file': '../OpenNMT/mlqa-de-en-dev.json',
#                         'output_dir': './out/squad-germanqa-bayes-finetune/',
                         'doc_stride': 128,
                         'version_2_with_negative': True,
                         'max_seq_length': 512,
                         'train_dataset': 'squad',
                         'dev_dataset': 'squad',
                         'overwrite_cache': False,
                         'max_query_length': 64,
                         'model_name_or_path': '../out/squad-roberta-base-10/best/pytorch_model.bin',
#                         'model_name_or_path': '../out/squadv1-roberta-base/best/pytorch_model.bin',
                         'lang': 'en',
                         'patience': 10})

if not os.path.exists(base_args.output_dir):
    os.makedirs(base_args.output_dir)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True,
                                             cache_dir='/ssd-playpen/home/adyasha/cache/')
train_dataset = load_and_cache_examples(base_args, tokenizer, evaluate=False, output_examples=False)
# base_args.dataset = 'newsqa'
dev_dataset, dev_examples, dev_features = load_and_cache_examples(base_args, tokenizer, evaluate=True,
                                                                  output_examples=True)


# from argparse import Namespace
# base_args = Namespace(**{'local_rank': -1,
#                          'train_file': '../AutoAdverse/data/train-sampled-v2.0.json',
#                          'predict_file': '../AutoAdverse/data/dev-split-v2.0.json',
#                          'output_dir': './out/squad-bayes/',
#                          'doc_stride': 128,
#                          'version_2_with_negative': True,
#                          'max_seq_length': 512,
#                          'train_dataset': 'squad',
#                          'dev_dataset': 'squad',
#                          'overwrite_cache': False,
#                          'max_query_length': 64,
#                          'model_name_or_path': '../out/squad-roberta-base-10/best/pytorch_model.bin'})
#
# if not os.path.exists(base_args.output_dir):
#     os.makedirs(base_args.output_dir)
#
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True, cache_dir='/ssd-playpen/home/adyasha/cache/')
# original_dataset = load_and_cache_examples(base_args, tokenizer, evaluate=False, output_examples=False)
# base_args.dataset = 'newsqa'
# dev_dataset, dev_examples, dev_features = load_and_cache_examples(base_args, tokenizer, evaluate=True, output_examples=True)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def roberta_train_and_eval(arg_string):

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--train_dataset", default="squad", type=str,
                        help="Dataset name, choices between SQuAD, NewsQA and HotPotQA, ")
    parser.add_argument("--dev_dataset", default="newsqa", type=str,
                        help="Dataset name, choices between SQuAD, NewsQA and HotPotQA, ")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--lang', default='en', type=str,
                        help="If lang != en, dataset is multilingual and needs alignment before evaluation")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_ratio", default=0, type=float,
                        help="Warmup over warmup_ratio*total_steps steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--patience', type=int, default=5,
                        help="Patience during evaluation.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args(arg_string)

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
                
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    model, _ = initializeRobertaForQA(args)
    model.to(args.device)

    # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
    # dev_dataset, dev_examples, dev_features = load_and_cache_examples(args, tokenizer, evaluate=True,
    #                                                                   output_examples=True)
    # global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # train_dataset = torch.utils.data.ConcatDataset([original_dataset, adversarial_dataset])
    _, _, results = trainRobertaForQA(args, train_dataset, model, tokenizer, (dev_dataset, dev_examples, dev_features))
    if results is None:
        results = evaluateRobertaForQA(args, model, tokenizer, dev_dataset, dev_examples, dev_features)

    logger.info("Results: {}".format(results))

    backup_log_file = os.path.join(args.output_dir, 'bayes-opt-log.txt')
    with open(backup_log_file, 'a+') as f:
        opt_args = {'args': arg_string,
                    'results': results}
        write_string = json.dumps(opt_args, indent=2)
        f.write(write_string)
        f.write('\n')
        
    del model

    return results['f1']

def roberta_wrapper(batch_size, warmup_ratio, learning_rate_multiplier, weight_decay):

    """Wrapper of RoBERTa training and validation.

    Notice how we ensure all parameters are casted
    to integer before we pass them along. Moreover, to avoid max_features
    taking values outside the (0, 1) range, we also ensure it is capped
    accordingly.
    """

    base_args_list = ['--model_type', 'roberta',
                      '--train_file', '../AutoAdverse/data/train-AutoAugment-QANet-SQuAD.json',
                      '--predict_file', '../AutoAdverse/data/dev-split-v2.0.json',
#                      '--train_file', '../AutoAdverse/data/train-BayesAugment-S2G.json',
#                      '--predict_file', '../OpenNMT/mlqa-de-en-dev.json',
                      '--train_dataset', 'squad',
                      '--dev_dataset', 'squad',
                      '--model_name_or_path', '../out/squad-roberta-base-10/best/pytorch_model.bin',
#                      '--model_name_or_path', '../out/squadv1-roberta-base/best/pytorch_model.bin',
#                      '--output_dir', './out/squad-germanqa-bayes-finetune/',
                      '--output_dir', './out/squad-bayes-AutoAugment',
                      '--logging_steps', '50',
                      '--do_train', '--do_eval', '--do_lower_case',
                      '--version_2_with_negative',
                      '--num_train_epochs', '2',
                      '--adam_epsilon', '1e-6',
                      '--max_seq_length', '512',
                      '--doc_stride', '128',
                      '--save_steps', '50',
                      '--overwrite_output_dir',
                      '--per_gpu_eval_batch_size', '6',
                      '--tokenizer_name', 'roberta-base',
                      '--config_name', 'roberta-base',
                      '--lang', 'en',
                      '--patience', '10']

    # compute per gpu batch size during training
    n_gpus = 2
    batch_size = 16 * batch_size
    if batch_size < 12: #8
        gradient_accumulation_steps = 1
        per_gpu_batch_size = 4
    elif batch_size >= 12 and batch_size < 20: #16
        gradient_accumulation_steps = 2
        per_gpu_batch_size = 4
    elif batch_size >=20 and batch_size < 28: #24
        gradient_accumulation_steps = 3
        per_gpu_batch_size = 4
    elif batch_size >= 28 and batch_size < 36: #32
        gradient_accumulation_steps = 4
        per_gpu_batch_size = 4
    else:
        raise NotImplementedError('%s batch size not implemented' % batch_size)
        
    per_gpu_batch_size = 6
    gradient_accumulation_steps = 1

    # learning rate in str
    learning_rate = str(round(learning_rate_multiplier, 1)) + 'e-5'

    opt_args_list = ['--learning_rate', learning_rate,
                     '--per_gpu_train_batch_size', str(per_gpu_batch_size),
                     '--gradient_accumulation_steps', str(gradient_accumulation_steps),
                     '--warmup_ratio', str(warmup_ratio),
                     '--weight_decay', str(round(weight_decay, 2))
                     ]

    return roberta_train_and_eval(base_args_list + opt_args_list)
    
def test_roberta():
    result = roberta_wrapper(batch_size=1.8997, warmup_ratio=0.2046, learning_rate_multiplier=1.0, weight_decay=0.1)

def optimize_roberta():
    """Apply Bayesian Optimization to SVC parameters."""

    optimizer = BayesianOptimization(
        f=roberta_wrapper,
        pbounds={"batch_size": (0.5, 2),
                 "warmup_ratio": (0.01, 0.5),
                 "learning_rate_multiplier": (1, 2),
                 "weight_decay": (0.01, 0.1)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(
        init_points=8,
        n_iter=32
        # What follows are GP regressor parameters
        # alpha=1e-3,
        # n_restarts_optimizer=5
    )
    print("Final result:", optimizer.max)


if __name__ == "__main__":

    print(Colours.yellow("--- Optimizing Roberta ---"))
    optimize_roberta()
    #test_roberta()
