import json, os
import spacy
from tqdm import tqdm
from termcolor import colored

nlp = spacy.load('de_core_news_sm')

TRANS_IN = {
    'de': '/ssd-playpen/home/adyasha/projects/fairseq/data/train.de',
    'jp': '',
    'fr': ''
}

TRANS_OUT = {
    'de': '/ssd-playpen/home/adyasha/projects/fairseq/data/output.en',
    'jp': '',
    'fr': ''
}

TRANS_ATTN = {
    'de': '/ssd-playpen/home/adyasha/projects/fairseq/data/output.en.attn.json',
    'jp': '',
    'fr': ''
}

ATTN_CONTEXT_MAP = {
    'de': '/ssd-playpen/home/adyasha/projects/fairseq/squad-de-en-context-map.json',
    'jp': '',
    'fr': '/ssd-playpen/home/adyasha/projects/data/mlqa/fr-dev-attn-context-map.json'
}

DATA_FILE = {'de': '/ssd-playpen/home/adyasha/projects/data/mlqa/dev-context-de-question-de.json',
              'fr': '/ssd-playpen/home/adyasha/projects/data/mlqa/mlqa-fr-en-dev.json'}

def read_translations(infile):

    with open(infile, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def get_bpe_replace_pos(tokens):

    bpe_pos = [i for i in range(len(tokens)) if '@@' in tokens[i]]

    replace_pos_groups = []
    cur_group = []
    for i in range(len(tokens)):
        if i in bpe_pos:
            cur_group.append(i)
        else:
            if cur_group:
                cur_group.append(i)
                replace_pos_groups.append(cur_group)
                cur_group = []
            else:
                replace_pos_groups.append(i)

    if cur_group:
        replace_pos_groups.append(cur_group)

    return replace_pos_groups

    
def fold_norm_attn_mat(attn_mat, src_tokens, tgt_tokens):

    assert len(attn_mat) == len(tgt_tokens) + 1
    assert all([len(row) == len(src_tokens) + 1 for row in attn_mat])

    src_replace_groups = get_bpe_replace_pos(src_tokens)
    tgt_replace_groups = get_bpe_replace_pos(tgt_tokens)

    updated_mat = []
    for pos in tgt_replace_groups:
        # print(pos)
        if isinstance(pos, int):
            updated_mat.append(attn_mat[pos])
        elif isinstance(pos, list):
            new_row = [sum([attn_mat[p][i] for p in pos]) for i in range(len(src_tokens)+1)]
            # print(new_row)
            total_prob = sum(new_row)
            new_row = [e/total_prob for e in new_row]
            updated_mat.append(new_row)
        else:
            raise ValueError

    # print(updated_mat)

    final_mat = [[] for _ in range(len(tgt_replace_groups))]
    for pos in src_replace_groups:
        # print(pos)
        if isinstance(pos, int):
            for i in range(len(updated_mat)):
                # print(pos, i)
                final_mat[i].append(updated_mat[i][pos])
        elif isinstance(pos, list):
            for i in range(len(updated_mat)):
                final_mat[i].append(sum([updated_mat[i][p] for p in pos]))
        else:
            raise ValueError

    print(src_replace_groups, len(src_replace_groups))
    print(tgt_replace_groups, len(tgt_replace_groups))
    print([' '.join([src_tokens[p] for p in group]) if isinstance(group, list) else src_tokens[group] for group in src_replace_groups])
    print([' '.join([tgt_tokens[p] for p in group]) if isinstance(group, list) else tgt_tokens[group] for group in tgt_replace_groups])

    return final_mat


def read_attn_file(infile):

    with open(infile, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    src_sentence = []
    tgt_sentence = []
    attn_mat = []
    src_tgt_mat_tuples = []

    for i, line in tqdm(enumerate(lines)):

        #print(line)

        tokens = line.strip().split()
        if len(tokens) == 1:
            assert len(attn_mat) == len(tgt_sentence), (src_sentence, tgt_sentence, attn_mat)
            src_tgt_mat_tuples.append((src_sentence, tgt_sentence, attn_mat))
            src_sentence = tokens
            n_words_src = len(src_sentence)
            tgt_sentence = []
            attn_mat = []
            continue

        if src_sentence == []:
            src_sentence = tokens
            n_words_src = len(src_sentence)
            continue

        if src_sentence != []:
            try:
                probs = [float(num[1:]) if num.startswith('*') else float(num) for num in tokens[1:]]
                assert len(probs) == n_words_src and all([p<=1.0 for p in probs]), (n_words_src, i, len(probs), line, src_sentence)
                tgt_sentence.append(tokens[0])
                attn_mat.append(probs)
            except (ValueError, AssertionError) as e:
                assert len(attn_mat) == len(tgt_sentence), (src_sentence, tgt_sentence, attn_mat)
                src_tgt_mat_tuples.append((src_sentence, tgt_sentence, attn_mat))
                src_sentence = tokens
                n_words_src = len(src_sentence)
                tgt_sentence = []
                attn_mat = []
                
    #src_tgt_mat_tuples.append((src_sentence, tgt_sentence, attn_mat))

    return src_tgt_mat_tuples

def get_attn_context_map(src_tgt_mat_tuples, translations):

    with open('../data/mlqa/dev-context-de-question-de.json', 'r') as f:
        dataset = json.load(f)

    attn_context_map = {}

    counter = 0
    for i, article in enumerate(dataset['data']):

        for j, paragraph in enumerate(article['paragraphs']):

            n_qas = len(paragraph['qas'])
            counter += n_qas
            context = paragraph['context']
            doc = nlp(context.strip())
            n_sents = len(list(doc.sents))
        
            attn_mats = [list(t) for t in src_tgt_mat_tuples[counter:counter+n_sents]]
            
            for s, (src_tokens, tgt_tokens, attn_mat) in zip(doc.sents, attn_mats):
                doc_tokens = [t.text for t in doc[s.start:s.end] if not t.text.isspace()]
                text = ' '.join(doc_tokens).lower()
                src_sentence = ' '.join(src_tokens)
                assert len(src_tokens) == len(doc_tokens), (text, src_sentence, len(doc_tokens), len(src_tokens))
                
            original_translations = translations[counter:counter+n_sents]
            for i, trans in enumerate(original_translations):
                trans_tokens = trans.split()
                if attn_mats[i][1][-1] == '</s>':
                    assert len(trans_tokens) == len(attn_mats[i][1]) - 1, (trans_tokens, attn_mats[i][1])
                    attn_mats[i][1] = trans_tokens + attn_mats[i][1][-1:]
                else:
                    assert len(trans_tokens) == len(attn_mats[i][1]), (trans_tokens, attn_mats[i][1])
                    attn_mats[i][1] = trans_tokens
                
            for qa in paragraph['qas']:
                attn_context_map[qa['id']] = attn_mats

            counter += n_sents

    return attn_context_map


def get_attn_context_map_transformer(trans_in, trans_out, attn_mat_dict):

    with open('../data/mlqa/dev-context-de-question-de.json', 'r') as f:
        dataset = json.load(f)

    attn_context_map = {}

    counter = 0
    for i, article in enumerate(dataset['data']):

        for j, paragraph in enumerate(article['paragraphs']):

            n_qas = len(paragraph['qas'])
            counter += n_qas
            context = paragraph['context']
            doc = nlp(context.strip())
            n_sents = len(list(doc.sents))

            attn_mats = [attn_mat_dict[str(counter+i)] for i in range(n_sents)]
            new_attn_mats = []

            for src, tgt, attn_mat in zip(trans_in[counter:counter+n_sents], trans_out[counter:counter+n_sents], attn_mats):

                src_tokens = src.split()
                tgt_tokens = tgt.split()

                new_attn_mat = fold_norm_attn_mat(attn_mat, src_tokens, tgt_tokens)
                # print(new_attn_mat)

                new_src_tokens = ' '.join(src_tokens).replace('@@ ', '').replace('@@', '').split()
                new_tgt_tokens = ' '.join(tgt_tokens).replace('@@ ', '').replace('@@', '').split()
                assert len(new_attn_mat) == len(new_tgt_tokens), (tgt_tokens, new_tgt_tokens, len(attn_mat), len(new_attn_mat))
                assert all([len(row) == len(new_src_tokens) for row in new_attn_mat]), (src_tokens, new_src_tokens, [len(row) for row in new_attn_mat], len(new_src_tokens), len(src_tokens))

                new_attn_mats.append((new_src_tokens, new_tgt_tokens, new_attn_mat))

            for qa in paragraph['qas']:
                attn_context_map[qa['id']] = new_attn_mats

            counter += n_sents

    return attn_context_map

def get_sublist_start(a, b):

    if not a: return True
    if not b: return False

    for i in range(len(b)-len(a) + 1):
        if b[i:i+len(a)] == a:
            return i
    return -1


def align_attention_score(prediction, src_tokens, tgt_tokens, attn_scores):

    prediction_tokens = prediction.split()
    start_idx = get_sublist_start(prediction_tokens, tgt_tokens)
    if start_idx == -1:
        print(colored("Did not find prediction in translation tokens", 'red'))
        print(prediction, tgt_tokens)
        return None

    scores = {}
    for i in range(len(src_tokens)):
        for j in range(i+1, len(src_tokens)):
        
            #print(i, j)

            common_n = 0.0
            recall_d, prec_d = 0.0, 0.0
            # precision denominator
            for m in range(i, j):
                prec_d += sum([s[m] for s in attn_scores])
                
            # recall demonimator
            for k in range(start_idx, start_idx+len(prediction_tokens)):
                recall_d += sum(attn_scores[k])
                
            # common numerator
            for k in range(start_idx, start_idx+len(prediction_tokens)):
                for m in range(i, j):
                    common_n += attn_scores[k][m]

            try:
                precision = float(common_n)/float(prec_d)
                recall = float(common_n)/float(recall_d)
                f1 = (2*precision*recall)/(precision+recall)
                #print(precision, recall, f1)
                scores[(i, j)] = f1
            except ZeroDivisionError:
                print(colored('ZERO DIVISION ERROR', 'red'))
                print(prediction, src_tokens, tgt_tokens)
                for a in attn_scores:
                    print(a)

    max_f1 = 0.0
    
    if scores == {}:
        return None
    
    #print(scores)
    for key, val in scores.items():
        if val > max_f1:
            max_f1 = val
            span = key

    #return ' '.join(src_tokens[span[0]:span[1]])
    return span


def get_aligned_spans(prediction_file, attn_context_map, lang):

    datafile = DATA_FILE[lang]
    with open(datafile, 'r') as f:
        dataset = json.load(f)

    with open(prediction_file, 'r') as f:
        data = json.load(f)

    new_predictions = {}

    not_found = 0
    for key, prediction in data.items():
        print(colored('Mapping for %s' % key, 'green'))

        #prediction = data[key]
        attn_mat_list = attn_context_map[key]
        if prediction == 'empty' or prediction == '':
            new_predictions[key] = ''
            print(colored('Skipped because empty prediction', 'blue'))
            continue
            
        foundContext = False
        for article in dataset['data']:
            for paragraph in article['paragraphs']:
                if key in [q['id'] for q in paragraph['qas']]:
                    context = paragraph['context']
                    foundContext = True
                
                if foundContext:
                    break
            if foundContext:
                break
       
        assert foundContext
        doc = nlp(context)

        foundSpan = False
        
        for (src_tokens, tgt_tokens, attn_mat), s in zip(attn_mat_list, doc.sents):
        
            tgt = ' '.join(tgt_tokens)
            original_tokens = [t.text for t in doc[s.start:s.end] if not t.text.isspace()]
            if prediction in tgt:
                aligned_span = align_attention_score(prediction, src_tokens, tgt_tokens, attn_mat)
                if aligned_span:
                    foundSpan = True
                    break
            else:
                continue
                
        if not foundSpan:
            not_found += 1
            tgt_tokens = [tgt for _, tgt, _ in attn_mat_list]
            tgt_context = ' '.join([' '.join(tokens) for tokens in tgt_tokens])
            print(colored('Prediction not found in target context?', 'red'))
            print(colored('Prediction: %s' % prediction, 'blue'))
            print(colored('Target context: %s' % tgt_context, 'blue'))
            new_predictions[key] = ''
        else:
            new_predictions[key] = ' '.join(original_tokens[aligned_span[0]:aligned_span[1]])
            print('%s mapped to %s' % (prediction, new_predictions[key]))
        
        
    print("%s predictions could not be mapped" % not_found)

    return new_predictions


def clean_translations(translations, remove_bpe=False):
    translations = [trans.replace('&apos;', "'") for trans in translations]
    translations = [trans.replace('&quot;', '"') for trans in translations]
    if remove_bpe:
        translations = [trans.replace('@@ ', '') for trans in translations]
    return translations
    
def read_attn_context_map(map_file):
    with open(map_file, 'r') as f:
        attn_context_map = json.load(f)
    return attn_context_map
    
def align_transformer_translations(pivot_lang_prediction_file, lang='de'):

    if not os.path.exists(ATTN_CONTEXT_MAP[lang]):
        with open(TRANS_ATTN[lang], 'r') as f:
            attn_mats = json.load(f)

        translation_inputs = read_translations(TRANS_IN[lang])
        translations = read_translations(TRANS_OUT[lang])
        translations = clean_translations(translations, False)
        assert len(translations) == len(attn_mats), (len(translations), len(attn_mats))

        attn_context_map = get_attn_context_map_transformer(translation_inputs, translations, attn_mats)
        with open(ATTN_CONTEXT_MAP[lang], 'w') as f:
            json.dump(attn_context_map, f)

    else:
        attn_context_map = read_attn_context_map(ATTN_CONTEXT_MAP[lang])

    new_predictions = get_aligned_spans(pivot_lang_prediction_file, attn_context_map, lang)

    outfile = pivot_lang_prediction_file.replace('predictions', 'predictions_' + lang)
    with open(outfile, 'w') as f:
       json.dump(new_predictions, f)

    return outfile

if __name__ == "__main__":

    #stm = read_attn_file('dev-de-sentences-attn.txt')
    
    #with open('src_tgt_attn.txt', 'w') as f:
    #    json.dump(stm, f)

    #translations = read_translations('dev-de-sentences-en.txt')

    #assert len(translations) == len(stm), (len(translations), len(stm))

    #attn_context_map = get_attn_context_map(stm, translations)

    #new_predictions = get_aligned_spans('/ssd-playpen/home/adyasha/projects/out/squad-roberta-base-10/eval-mlqa/AddSentDiverse/predictions_.json', attn_context_map)
    
    #with open('new_predictions_AddSentDiverse.json', 'w') as f:
    #    json.dump(new_predictions, f)
    
    src_tokens = ["was", "war", "die", "positio", "der", "berline", "gericht", "gegen\u00fcb", "gro\u00dfbri", "und", "russlan", "?"]
    tgt_tokens = ["what", "was", "the", "position", "of", "the", "berlin", "court", "to", "the", "u.k.", "and", "russia", "?", "</s>"]
    attn_mat = [[0.4195063, 0.1225115, 0.0765876, 0.0754822, 0.0229257, 0.0133459, 0.0021495, 0.0190706, 0.0106536, 0.0083552, 0.0065206, 0.2228914], [0.0517764, 0.5080844, 0.211239, 0.1148611, 0.0589948, 0.0076464, 0.0012273, 0.0047265, 0.002443, 0.0012841, 0.0016072, 0.03611], [0.0279904, 0.0517525, 0.454592, 0.1553895, 0.2531115, 0.0397388, 0.0024832, 0.0055084, 0.0052167, 0.0003879, 0.0005611, 0.0032678], [0.0082777, 0.0092387, 0.0251159, 0.514525, 0.0414525, 0.3881996, 0.0049141, 0.001426, 0.0056926, 0.0001809, 8.28e-05, 0.0008941], [0.0249529, 0.0505431, 0.0809695, 0.1980994, 0.370602, 0.1940006, 0.0287312, 0.0340603, 0.0026484, 0.004075, 0.0002986, 0.011019], [0.0128451, 0.0182277, 0.0260035, 0.1059609, 0.3164611, 0.4980354, 0.0069171, 0.004702, 0.0065517, 0.0007918, 0.0015844, 0.0019194], [0.0003311, 0.0013212, 0.000931, 0.0362578, 0.0098777, 0.9391688, 0.0055532, 0.0020395, 0.0042401, 8.84e-05, 2.67e-05, 0.0001644], [0.0003596, 0.0007548, 0.0005824, 0.0070361, 0.0029597, 0.0050707, 0.9658388, 0.0102032, 0.0011335, 0.0052311, 1.4e-05, 0.0008161], [0.0098658, 0.0231743, 0.0182242, 0.0844691, 0.0319083, 0.0146287, 0.03594, 0.6615327, 0.0531291, 0.0333505, 0.0027085, 0.0310688], [0.0031487, 0.0073809, 0.0030823, 0.0225269, 0.0069527, 0.0077874, 0.0040793, 0.1422688, 0.7068807, 0.0365215, 0.0536462, 0.0057244], [8.73e-05, 0.0001313, 8.17e-05, 0.0025494, 0.0001094, 0.0055096, 0.0005571, 0.0032516, 0.9811095, 0.003344, 0.0026044, 0.0006649], [0.029473, 0.0304812, 0.014722, 0.056298, 0.0117261, 0.0092966, 0.0394061, 0.1218546, 0.0385965, 0.4621425, 0.0110307, 0.1749728], [0.0054503, 0.0053706, 0.0017942, 0.0067507, 0.0008928, 0.0015002, 0.0051071, 0.0162511, 0.0216599, 0.0636223, 0.8463793, 0.0252218], [0.0201387, 0.0287334, 0.0198956, 0.0616222, 0.0113876, 0.0062686, 0.0086559, 0.0295828, 0.0032531, 0.0134891, 0.0053135, 0.7916595], [0.1111758, 0.0878493, 0.0524435, 0.0695454, 0.0249774, 0.0178797, 0.0048278, 0.0248931, 0.0054334, 0.0167245, 0.0150586, 0.5691915]]
    prediction = "berlin court"
    
    tgt_tokens = ["auerdem", "wurde", "es", "mglich", ",", "die", "karten", "ohne", "ab@@", "sender", "zu", "versch@@", "icken", "."]
    src_tokens = ["It", "was", "also", "possible", "to", "send", "the", "cards", "without", "the", "sender", "."]
    prediction = "die karten"
    attn_mat = [[0.1916, 0.0586, 0.1202, 0.0204, 0.0208, 0.0065, 0.0091, 0.0050, 0.0068,
         0.0023, 0.0028, 0.0129, 0.0114],
        [0.0669, 0.0904, 0.0958, 0.1130, 0.0328, 0.0073, 0.0060, 0.0029, 0.0069,
         0.0035, 0.0025, 0.0164, 0.0066],
        [0.0569, 0.0282, 0.0373, 0.0470, 0.0157, 0.0035, 0.0028, 0.0011, 0.0052,
         0.0030, 0.0016, 0.0053, 0.0030],
        [0.0379, 0.0259, 0.0709, 0.0914, 0.0212, 0.0086, 0.0047, 0.0026, 0.0078,
         0.0019, 0.0023, 0.0043, 0.0024],
        [0.0300, 0.0161, 0.0296, 0.0326, 0.0349, 0.0159, 0.0069, 0.0029, 0.0136,
         0.0028, 0.0023, 0.0031, 0.0034],
        [0.0285, 0.0045, 0.0055, 0.0049, 0.0259, 0.0429, 0.0372, 0.0158, 0.0150,
         0.0082, 0.0051, 0.0057, 0.0049],
        [0.1363, 0.0100, 0.0111, 0.0102, 0.0273, 0.0322, 0.2850, 0.2660, 0.0304,
         0.0177, 0.0651, 0.0205, 0.0087],
        [0.0176, 0.0110, 0.0064, 0.0060, 0.0610, 0.0431, 0.0452, 0.0249, 0.1018,
         0.0498, 0.0306, 0.0427, 0.0083],
        [0.0039, 0.0017, 0.0020, 0.0020, 0.0045, 0.0105, 0.0093, 0.0114, 0.0203,
         0.0995, 0.0929, 0.0144, 0.0030],
        [0.0069, 0.0030, 0.0034, 0.0035, 0.0063, 0.0151, 0.0137, 0.0171, 0.0300,
         0.1763, 0.1640, 0.0282, 0.0045],
        [0.0115, 0.0048, 0.0102, 0.0087, 0.0140, 0.0683, 0.0067, 0.0032, 0.0121,
         0.0054, 0.0037, 0.0050, 0.0025],
        [0.0158, 0.0064, 0.0167, 0.0133, 0.0295, 0.0863, 0.0114, 0.0057, 0.0166,
         0.0074, 0.0056, 0.0096, 0.0036],
        [0.0161, 0.0065, 0.0145, 0.0109, 0.0286, 0.0906, 0.0099, 0.0053, 0.0136,
         0.0073, 0.0053, 0.0094, 0.0032],
        [0.0567, 0.0233, 0.0235, 0.0233, 0.0699, 0.0386, 0.0537, 0.0466, 0.0479,
         0.0581, 0.0647, 0.0791, 0.0484],
        [0.3235, 0.7098, 0.5529, 0.6127, 0.6077, 0.5305, 0.4985, 0.5894, 0.6721,
         0.5567, 0.5515, 0.7435, 0.8863]]

    # new_attn_mat = fold_norm_attn_mat(attn_mat, src_tokens, tgt_tokens)
    # print(new_attn_mat)
    #
    # new_src_tokens = ' '.join(src_tokens).replace('@@ ', '').split()
    # new_tgt_tokens = ' '.join(tgt_tokens).replace('@@ ', '').split()
    # assert len(new_attn_mat) == len(new_tgt_tokens)
    # assert all([len(row) == len(new_src_tokens) for row in new_attn_mat])
    # span = align_attention_score(prediction, new_src_tokens, new_tgt_tokens, new_attn_mat)
    #
    # print(new_src_tokens[span[0]:span[1]])

    # print(get_bpe_replace_pos(src_tokens))
    # print(get_bpe_replace_pos(tgt_tokens))

    align_transformer_translations('')

    # x = get_sublist_start(['him', 'on', '549', 'from', '640', 'in', 'India', '.'], ['That', 'puts', 'him', 'on', '549', 'from', '640', 'in', 'India', '.'])
    # print(x)

