import json
import spacy
from nltk import word_tokenize
import random

nlp = spacy.load('en_core_web_sm')

def create_dev_test(data_file):
    root = json.load(open(data_file, 'r'))
    samples = root['data']
    n_articles = len(samples)
    idxs = list(range(n_articles))
    random.shuffle(idxs)
    dev_idxs = idxs[:int(n_articles/2)]
    test_idxs = idxs[int(n_articles/2):]

    dev = [samples[idx] for idx in dev_idxs]
    test = [samples[idx] for idx in test_idxs]

    with open(data_file.replace('.json', '.dev.json'), 'w') as f:
        root['data'] = dev
        json.dump(root, f)
    with open(data_file.replace('.json', '.test.json'), 'w') as f:
        root['data'] = test
        json.dump(root, f)

def parse_alignments(infile, outfile):

    with open(infile, 'r') as f:
        input_data = f.readlines()
    with open(outfile, 'r') as f:
        output = f.readlines()

    alignments = {}
    for input_lines, out in zip(input_data, output):
        src_line, tgt_line = input_lines.split(' ||| ')
        alignment = {int(a.split('-')[0]):int(a.split('-')[1]) for a in out.split()}
        alignments[src_line] = alignment
    #print(alignments)
    return alignments

def get_word_idxs(doc, start, end):
    start_positions = [tok.idx for tok in doc]
    end_positions = [tok.idx+len(tok.text) for tok in doc]
    start_found = False
    end_found = False
    pred_idxs = []
    for i, (start_pos, end_pos) in enumerate(zip(start_positions, end_positions)):
        if start_found == False and start_pos >= start:
            start_found = True
            pred_idxs.append(i)
        elif start_pos >= end:
            break
        elif start_found ==True and end_found == False:
            pred_idxs.append(i)
        else:
            continue
    return pred_idxs

def align_predictions_to_tgt(predictions, data_file, alignments):

    ref_samples = json.load(open('../data/xquad/xquad.tr.dev.json', 'r'))["data"]
    ref_ids = []
    for article in ref_samples:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                ref_ids.append(qa["id"])

    samples = json.load(open(data_file, 'r'))
    predictions = json.load(open(predictions, 'r'))

    tgt_predictions = {}
    skipped = 0
    
    for id in predictions.keys():
        if id not in ref_ids:
            continue
        
        sample = samples[id]
        context = sample["context"]
        src_tokens = [tok.text for tok in nlp.tokenizer(sample["context"])]
        alignment = alignments[' '.join(src_tokens)]
        tgt_context = sample["tgt_context"]
        tgt_tokens = word_tokenize(sample["tgt_context"])

        start_idx = context.find(predictions[id])
        if start_idx == -1:
            print(context, predictions[id])
            skipped += 1
            continue
        end_idx = start_idx + len(predictions[id])
        doc = nlp(context)
        print(end_idx, start_idx)
        word_idxs = get_word_idxs(doc, start_idx, end_idx)

        print(alignment)
        print(context, tgt_tokens, predictions[id])
        print(word_idxs)
        target_word_idxs = [alignment[idx] for idx in word_idxs if idx in alignment]
        print(target_word_idxs)
        try:
            max_idx = max(target_word_idxs)
            min_idx = min(target_word_idxs)
        except:
            skipped += 1
            continue
        print(predictions[id], tgt_tokens[min_idx:max_idx+1])

        if len(target_word_idxs) == 1:
            tgt_prediction = tgt_tokens[min_idx]
        else:
            start_idx = tgt_context.find(tgt_tokens[min_idx])
            end_idx = tgt_context.find(tgt_tokens[max_idx], start_idx)
            tgt_prediction = tgt_context[start_idx:end_idx+len(tgt_tokens[max_idx])]
        tgt_predictions[id] = tgt_prediction

    print(skipped, len(tgt_predictions), len(predictions))

    with open('xquad.tr.en.dev.predictions.json', 'w') as f:
        json.dump(tgt_predictions, f)

#create_dev_test('../data/xquad/xquad.tr.json')
alignments = parse_alignments('../translate/data/xquad.tr.fast.align.input.txt', '../fast_align/build/reverse.align')
align_predictions_to_tgt('../out/squadv1-roberta-base/eval-mlqa/best/tr/predictions_.json', '../translate/data/xquad.tr.en.json', alignments)
