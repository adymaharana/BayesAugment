import codecs, csv, json
import argparse
from collections import defaultdict
import random
from termcolor import colored
from tqdm import tqdm

output_file = './out/squad_dev_input.out'
corenlp_cache = ''

import numpy as np

def cos_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def read_data(filename):
    with open(filename) as f:
        return json.load(f)


def load_cache(cache_file):
    with open(cache_file) as f:
        return json.load(f)

def process_paraphrase(paraphrase, answer_tokens):

    answer_words = [t['word'].lower() for t in answer_tokens]
    paraphrase_tokens = paraphrase.split()
    for i, t in enumerate(paraphrase_tokens):
        try:
            loc = answer_words.index(t.lower())
        except:
            continue
        original_word = answer_tokens[loc]['originalText']
        paraphrase_tokens[i] = original_word

    return ' '.join(paraphrase_tokens)


def extract_phrases(sentence_corenlp_tokens):

    phrases = []

    ner_keys = ['PERSON', 'LOCATION', 'ORGANIZATION', 'MISC', 'MONEY', 'NUMER', 'ORDINAL', 'PERCENT', 'DATE', 'TIME', 'DURATION', 'SET']
    for key in ner_keys:
        current_phrase =[]
        for i, token in enumerate(sentence_corenlp_tokens):
            if token['ner'] == key:
                current_phrase.append((i, token))
            else:
                if len(current_phrase) > 0:
                    phrases.append(current_phrase)
                    current_phrase = []

    POS_TYPES = {'Noun': ['NN', 'NNS', 'NNP', 'NNS'],
                 # 'Verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                 # 'Adverb': ['RB', 'RBR', 'RBS'],
                 'Adjective': ['JJ', 'JJR', 'JJS']}

    for key, val in POS_TYPES.items():
        current_phrase =[]
        for i, token in enumerate(sentence_corenlp_tokens):
            if token['pos'] in val:
                current_phrase.append((i, token))
            else:
                if len(current_phrase) > 0:
                    phrases.append(current_phrase)
                    current_phrase = []

    return phrases

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


def get_tokens_for_answers(answer_objs, corenlp_obj):
    """Get CoreNLP tokens corresponding to a SQuAD answer object."""
    first_a_toks = None
    a_idx = 0
    for i, a_obj in enumerate(answer_objs):
        a_toks = []
        answer_start = a_obj['answer_start']
        answer_end = answer_start + len(a_obj['text'])
        for cidx, s in enumerate(corenlp_obj['sentences']):
            for t in s['tokens']:
                if t['characterOffsetBegin'] >= answer_end: continue
                if t['characterOffsetEnd'] <= answer_start: continue
                a_toks.append(t)
                a_idx = cidx
        if corenlp_rejoin(a_toks).strip() == a_obj['text']:
            # Make sure that the tokens reconstruct the answer
            return i, a_toks, a_idx
        if i == 0: first_a_toks = a_toks
    # None of the extracted token lists reconstruct the answer
    # Default to the first
    return 0, first_a_toks, a_idx

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r',encoding='utf-8')
    model = {}
    for line in tqdm(f):
        row = line.strip().split(' ')
        word = row[0]
        #print(word)
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

def compare_paraphrase_heuristics(q_parse, qa_sample, context_parse, paraphrases):

    q_content_phrases = extract_phrases(q_parse['tokens'])
    ind, a_toks, ans_loc = get_tokens_for_answers(qa_sample['answers'], context_parse)

    answer_text = qa_sample['answers'][0]['text']
    print('Question: ', qa_sample['question'])
    print('Answer Sentence: ', corenlp_rejoin(context_parse['sentences'][ans_loc]['tokens']))
    print('Answer: ', answer_text)
    phrases = [' '.join([t['word'] for i, t in phrase]) for phrase in q_content_phrases]
    print(phrases)
    found = False
    correct_paraphrases = []
    for paraphrase in paraphrases:
        if answer_text.lower() in paraphrase:
            num_phrases_overlap = sum([phrase.lower() in paraphrase for phrase in phrases])
            if num_phrases_overlap >= len(phrases) * thresh:
                correct_paraphrases.append(paraphrase)
                print(paraphrase)
                found = True
                final_count += 1
    return correct_paraphrases

def compare_paraphrase_glove(q_parse, paraphrases, gloveModel):

    q_words = [word['word'].lower() for word in q_parse['tokens']]
    q_embeddings = [gloveModel[word] for word in q_words if word in gloveModel]
    
    similarities = []
    for i, paraphrase in enumerate(paraphrases):
        p_words = paraphrase.split()

        p_embeddings = [gloveModel[word] for word in p_words if word in gloveModel]
        if q_embeddings == [] or p_embeddings == []:
            continue
        
        q_embedding = np.average(q_embeddings, axis=0)
        p_embedding = np.average(p_embeddings, axis=0)

        cossim = cos_sim(p_embedding, q_embedding)
        similarities.append((i, cossim))

    if similarities == []:
        return None
        
    ranked_sims = sorted(similarities, key=lambda t: t[1], reverse=True)

    ranked_paraphrases = [(paraphrases[i], sim) for i, sim in ranked_sims]

    return ranked_paraphrases


def process_squad_out_data(args):

    gloveModel = loadGloveModel(args.glove_file)

    corenlp_cache = load_cache(args.corenlp_cache_file)
    dataset = read_data(args.dataset_file)

    # read parsed data
    outfile = codecs.open(args.output_file, 'r', 'utf-8')
    out_reader = csv.DictReader(outfile, delimiter='\t')

    # loop over sentences and transform them
    paraphrase_dict = defaultdict(lambda: [])
    for d_idx, ex in enumerate(out_reader):
        if ex['template'] == 'GOLD':
            continue
        paraphrase_dict[ex['idx']].append(ex['sentence'])

    final_perturbation_dict = {}

    num_questions = 0
    p_count = 0
    num_q_parsed = 0
    final_count = 0
    thresh = 0.3
    for article in dataset['data']:
        for i, paragraph in enumerate(article['paragraphs']):
            context = paragraph['context']
            context_parse = corenlp_cache[context]
            for j, qa_sample in enumerate(paragraph['qas']):
                if qa_sample['is_impossible']:
                    continue
                num_questions += 1
                try:
                    q_parse = corenlp_cache[qa_sample['question']]
                except KeyError:
                    continue
                idx = '%s-%s-%s' % (article['title'], i, j)
                if idx not in paraphrase_dict:
                    continue
                paraphrases = paraphrase_dict[idx]
                correct_paraphrases = compare_paraphrase_glove(q_parse, paraphrases, gloveModel)
                print(qa_sample["question"])
                if correct_paraphrases is None:
                    print("No paraphrases found")
                    continue
                else:
                    print("Ranked Paraphrases")
                    for paraphrase in correct_paraphrases:
                        print(colored(paraphrase[0], 'blue'), colored(paraphrase[1], 'red'))

                candidate_paraphrases = correct_paraphrases[0:3]
                # processed_paraphrase = process_paraphrase(candidate_paraphrase[0][0], context_parse['sentences'][ans_loc]['tokens'])

                # if found:
                num_q_parsed += 1
                final_perturbation_dict[qa_sample['id']] = candidate_paraphrases

                # print(colored(candidate_paraphrase, 'blue'))
                # print(colored(processed_paraphrase, 'green'))

                # print('\n')

    print(num_questions, num_q_parsed, p_count, final_count)

    with open('out/squad_train_syntactic_paraphrases.json', 'w') as f:
        json.dump(final_perturbation_dict, f)
        
def process_newsqa_out_data(args):

    gloveModel = loadGloveModel(args.glove_file)

    corenlp_cache = load_cache(args.corenlp_cache_file)
    dataset = read_data(args.dataset_file)

    # read parsed data
    outfile = codecs.open(args.output_file, 'r', 'utf-8')
    out_reader = csv.DictReader(outfile, delimiter='\t')

    # loop over sentences and transform them
    paraphrase_dict = defaultdict(lambda: [])
    for d_idx, ex in enumerate(out_reader):
        if ex['template'] == 'GOLD':
            continue
        paraphrase_dict[ex['idx']].append(ex['sentence'])

    final_perturbation_dict = {}

    num_questions = 0
    p_count = 0
    num_q_parsed = 0
    final_count = 0
    thresh = 0.3
    for story in dataset['data']:
        for i, qa_sample in enumerate(story['questions']):
            num_questions += 1
            try:
                q_parse = corenlp_cache[qa_sample['q']]
            except KeyError:
                print("Not found in context")
                continue
            idx = '%s-%s' % (story['storyId'], i)
            if idx not in paraphrase_dict:
                continue
            paraphrases = paraphrase_dict[idx]
            correct_paraphrases = compare_paraphrase_glove(q_parse, paraphrases, gloveModel)
            print(qa_sample["q"])
            if correct_paraphrases is None:
                print("No paraphrases found")
                continue
            else:
                print("Ranked Paraphrases")
                for paraphrase in correct_paraphrases:
                    print(colored(paraphrase[0], 'blue'), colored(paraphrase[1], 'red'))

            candidate_paraphrases = correct_paraphrases[0:3]
            # processed_paraphrase = process_paraphrase(candidate_paraphrase[0][0], context_parse['sentences'][ans_loc]['tokens'])

            # if found:
            num_q_parsed += 1
            final_perturbation_dict[idx] = candidate_paraphrases

            # print(colored(candidate_paraphrase, 'blue'))
            # print(colored(processed_paraphrase, 'green'))

            # print('\n')

    print(num_questions, num_q_parsed, p_count, final_count)

    with open('out/newsqa_dev_syntactic_paraphrases.json', 'w') as f:
        json.dump(final_perturbation_dict, f)

def main():

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument('--output_file', type=str, help='Path to output file generated by SCPN')
    parser.add_argument('--corenlp_cache_file', type=str)
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--glove_file', type=str)
    parser.add_argument('--dataset_type', type=str, default='squad')
    
    args = parser.parse_args()
    
    if args.dataset_type == 'squad':
        process_squad_out_data(args)
    else:
        process_newsqa_out_data(args)

if __name__ == "__main__":
    main()