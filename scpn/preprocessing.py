import json
import pandas as pd

def corenlp_rejoin(tokens, sep=None):
    """Rejoin tokens into the original sentence.

    Args:
      tokens: a list of dicts containing 'originalText' and 'before' fields.
          All other fields will be ignored.
      sep: if provided, use the given character as a separator instead of
          the 'before' field (e.g. if you want to preserve where tokens are).
    Returns: the original sentence that generated this CoreNLP token list.
    """
    if sep is None:
        return ''.join('%s%s' % (t['before'], t['originalText']) for t in tokens)
    else:
        # Use the given separator instead
        return sep.join(t['originalText'] for t in tokens)

def load_corenlp_cache(cache_file):

    with open(cache_file, 'r') as f:
        corenlp_cache = json.load(f)

    return corenlp_cache

def load_dataset(dataset_file):

    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    return dataset

def dump_scpn_squad_input(cache_file, dataset_file):

    corenlp_cache = load_corenlp_cache(cache_file)
    dataset = load_dataset(dataset_file)
    ids, sentences, parses = [], [], []
    skipped = 0

    for article in dataset['data']:
        for i, paragraph in enumerate(article['paragraphs']):

            context_parse = corenlp_cache[paragraph['context']]
            try:
                for j, sentence in enumerate(context_parse['sentences']):
                    parse = sentence['parse']
                    id = "%s-%s-%s" % (article['title'], i, j)
                    sentence_text = ' '.join([word['word'] for word in sentence['tokens']])
                    parse = parse.replace('\n', '')
                    parse = parse.replace('-TMP', '')
                    ids.append(id)
                    sentences.append(sentence_text)
                    parses.append(parse)
            except:
                # print(article, paragraph['context'], context_parse)
                skipped += 1

    df = pd.DataFrame({'idx': ids, 'tokens': sentences, 'parse': parses})
    df.to_csv('data/squad_train_input.tsv', sep='\t')
    print("%s paragraphs not found in corenlp cache" % skipped)

def dump_scpn_newsqa_input(cache_file, dataset_file):

    corenlp_cache = load_corenlp_cache(cache_file)
    dataset = load_dataset(dataset_file)
    ids, sentences, parses = [], [], []
    skipped = 0

    for story in dataset['data']:
        try:
            for i, qa in enumerate(story['questions']):
                question = qa['q']
                corenlp_obj = corenlp_cache[question]
                parse = corenlp_obj['parse']
                id = "%s-%s" % (story['storyId'], i)
                sentence_text = ' '.join([word['word'] for word in corenlp_obj['tokens']])
                parse = parse.replace('\n', '')
                parse = parse.replace('-TMP', '')
                ids.append(id)
                sentences.append(sentence_text)
                parses.append(parse)
        except KeyError:
            # print(article, paragraph['context'], context_parse)
            skipped += 1

    df = pd.DataFrame({'idx': ids, 'tokens': sentences, 'parse': parses})
    df.to_csv('data/newsqa_ques_train_input.tsv', sep='\t')
    print("%s paragraphs not found in corenlp cache" % skipped)

#dump_scpn_input('../AutoAdverse/data/dev-split-v2.0_const_parse.json', '../AutoAdverse/data/dev-split-v2.0.json')
dump_scpn_newsqa_input('../data/newsqa/train_corenlp_cache.json', '../data/newsqa/train.json')