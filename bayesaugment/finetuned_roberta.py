from torch.nn import CrossEntropyLoss, KLDivLoss, BCEWithLogitsLoss
from torch import nn

from transformers import (WEIGHTS_NAME, BertConfig, BertPreTrainedModel,
                                  BertForQuestionAnswering,
                                  BertTokenizer, RobertaTokenizer,
                                  RobertaConfig, RobertaTokenizer, RobertaModel)



from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
# WarmupLinearSchedule

import torch
from tqdm import tqdm, trange
import random
import numpy as np
import os, json
import logging
logger = logging.getLogger(__name__)

from squad_roberta import RawResult, RawResultExtended, write_predictions, write_predictions_extended
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_predict_file
# from utils_squad_multilingual import align_transformer_translations, DATA_FILE
# from mlqa_evaluation_v1 import EVAL_OPTS_MLQA, main as evaluate_on_mlqa
# from squad_evaluate_official import evaluate_squad_predictions

from modelling_distilroberta import DistilRobertaForQuestionAnswering

logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.ERROR)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class RobertaForQuestionAnswering(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss, start_scores, end_scores = outputs[:2]

    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        print('Hidden Size: %s' % config.hidden_size)

        self.version_2_with_negative = True

        if self.version_2_with_negative:
            self.clf_output = PoolerAnswerClass(config.hidden_size)

        #self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None, binary_targets=None):

        # print(input_ids.size(),
        #       attention_mask.size())
        # print(token_type_ids,
        #       position_ids,
        #       head_mask)
        # print(start_positions.size(),
        #       end_positions.size(),
        #       binary_targets.size())

        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if self.version_2_with_negative:
            clf_logits = self.clf_output(sequence_output)
            outputs = (start_logits, end_logits, clf_logits,) + outputs[2:]
        else:
            outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if self.version_2_with_negative and binary_targets is not None:
                clf_loss_fct = BCEWithLogitsLoss()
                clf_loss = clf_loss_fct(clf_logits, binary_targets)
                total_loss += clf_loss

            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class PoolerAnswerClass(nn.Module):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """
    def __init__(self, hidden_size, dropout=0.2):
        super(PoolerAnswerClass, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, int(hidden_size/2))
        self.activation = nn.Tanh() #Mish() # nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.dense_1 = nn.Linear(int(hidden_size/2), 1)

    def forward(self, hidden_states, cls_index=None):

        # if cls_index is not None:
        #     cls_index = cls_index[:, None, None].expand(-1, -1, hsz) # shape (bsz, 1, hsz)
        #     cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2) # shape (bsz, hsz)
        # else:
        cls_token_state = hidden_states[:, 0, :] # shape (bsz, hsz)

        x = self.dense_0(cls_token_state)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense_1(x).squeeze(-1)

        return x

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    'distilroberta-base': (RobertaConfig, DistilRobertaForQuestionAnswering, RobertaTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initializeRobertaForQA(args):

    logger.info("Initializing pre-trained RoBERTa model")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir='/ssd-playpen/home/adyasha/cache/')
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir='/ssd-playpen/home/adyasha/cache/')
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir='/ssd-playpen/home/adyasha/cache/')
    model.to(args.device)

    return model, tokenizer

def trainRobertaForQA(args, train_dataset, model, tokenizer=None, dev_data=None, global_max_f1=0):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(t_total*args.warmup_ratio), t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total*args.warmup_ratio), num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    results_history = []
    stop_training = False
    max_f1 = None

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type in ['xlm', 'roberta', 'distilroberta-base'] else batch[2],
                      'start_positions': batch[3],
                      'end_positions': batch[4]}
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5],
                               'p_mask': batch[6]})
            if args.version_2_with_negative:
                inputs.update({'binary_targets': batch[7]})

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    if tokenizer is not None and dev_data is not None:
                        dev_dataset, dev_examples, dev_features = dev_data
                        results = evaluateRobertaForQA(args, model, tokenizer, dev_dataset, dev_examples, dev_features)
                        results_history.append(results)
                        max_results = sorted(results_history, key=lambda d: d['f1'])[-1]
                        print("Maximum F1 score so far = ", max_results['f1'])
                        # Patience = 5 logging steps
                        if len(results_history) > args.patience and max_results not in results_history[-args.patience:]:
                            stop_training = True

                        # for key, value in results.items():
                        #     tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    # tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                #if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                if  global_step % args.logging_steps == 0:
                
                    if max_results['f1'] != results['f1']:
                        continue
                   
                    # Save best model checkpoint
                    if max_results['f1'] >= global_max_f1:
                        #rand_step = '%s%s%s' % (random.randint(0, 10), random.randint(0, 10), random.randint(0, 10))
                        rand_step = str(round(max_results['f1'], 2))
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(rand_step))
                        with open(os.path.join(args.output_dir, 'results_history.txt'), 'a+') as fres:
                            fres.write('checkpoint-{}'.format(rand_step) + ': ' + json.dumps(max_results))
                    else:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            
            if stop_training:
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if stop_training:
            train_iterator.close()
            break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()
    logging.info("Returning %s" % max_f1)
    return global_step, tr_loss / global_step, max_results


def evaluateRobertaForQA(args, model, tokenizer, dataset, examples, features, data_file= None, prefix=""):

    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type in ['xlm', 'roberta', 'distilroberta-base'] else batch[2]  # XLM don't use segment_ids
                      }
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id            = unique_id,
                                           start_top_log_probs  = to_list(outputs[0][i]),
                                           start_top_index      = to_list(outputs[1][i]),
                                           end_top_log_probs    = to_list(outputs[2][i]),
                                           end_top_index        = to_list(outputs[3][i]),
                                           cls_logits           = to_list(outputs[4][i]))
            else:
                if args.version_2_with_negative:
                    result = RawResult(unique_id=unique_id,
                                       start_logits=to_list(outputs[0][i]),
                                       end_logits=to_list(outputs[1][i]),
                                       clf_logit=to_list(outputs[2][i]))
                else:
                    result = RawResult(unique_id    = unique_id,
                                       start_logits = to_list(outputs[0][i]),
                                       end_logits   = to_list(outputs[1][i]))
            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    if args.model_type in ['xlnet', 'xlm']:
        # XLNet uses a more complex post-processing procedure
        write_predictions_extended(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.predict_file,
                        model.config.start_n_top, model.config.end_n_top,
                        args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        write_predictions(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)

    evaluate_options = EVAL_OPTS(data_file=args.predict_file if data_file is None else data_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file,
                                 dataset=args.dev_dataset)
    results = evaluate_on_predict_file(evaluate_options)
                        
    # if args.lang in ['de', 'fr']:
    #     output_prediction_file = align_transformer_translations(output_prediction_file, lang=args.lang)
    #     data_file = DATA_FILE[args.lang]
    # print(output_prediction_file, data_file)

    # Evaluate with the official SQuAD script
    # if args.lang == 'en' or args.lang == 'de':
    #     evaluate_options = EVAL_OPTS(data_file=args.predict_file if data_file is None else data_file,
    #                                  pred_file=output_prediction_file,
    #                                  na_prob_file=output_null_log_odds_file,
    #                                  dataset=args.dev_dataset)
    #     results = evaluate_on_predict_file(evaluate_options)
    # else:
    #     evaluate_options = EVAL_OPTS_MLQA(dataset_file=args.predict_file,
    #                                       prediction_file=output_prediction_file,
    #                                       answer_language=args.lang)
    #     results = evaluate_on_mlqa(evaluate_options)

    return results