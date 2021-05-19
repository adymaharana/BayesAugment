"""Variety of tools regarding the AddSent adversary."""
import argparse
import collections
import json
import math
from nectar.nectar import corenlp
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer
# import nltk
# nltk.data.path.append('/ssd-playpen/home/adyasha/nltk_data')
from pattern import en as patten
import random
from termcolor import colored
import sys
import os
import logging
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)

OPTS = None

QUIET = True

STEMMER = LancasterStemmer()

POS_TO_WORDNET = {
    'NN': wn.NOUN,
    'NNS': wn.NOUN,
    'JJ': wn.ADJ,
    'JJR': wn.ADJ,
    'JJS': wn.ADJ,
    'RB': wn.ADV,
    'RBR': wn.ADV,
    'RBS': wn.ADV,
    'VB': wn.VERB,
    'VBD': wn.VERB,
    'VBG': wn.VERB,
    'VBZ': wn.VERB,
    'VBN': wn.VERB,
    'VBP': wn.VERB
}

# Map to pattern.en aliases
# http://www.clips.ua.ac.be/pages/pattern-en#conjugation
POS_TO_PATTERN = {
    'vb': 'inf',  # Infinitive
    'vbp': '1sg',  # non-3rd-person singular present
    'vbz': '3sg',  # 3rd-person singular present
    'vbg': 'part',  # gerund or present participle
    'vbd': 'p',  # past
    'vbn': 'ppart',  # past participle
}
# Tenses prioritized by likelihood of arising
PATTERN_TENSES = ['inf', '3sg', 'p', 'part', 'ppart', '1sg']

CORENLP_LOG = 'corenlp.log'
CORENLP_PORT = 8101
COMMANDS = ['print-questions', 'print-answers', 'corenlp', 'convert-q',
            'inspect-q', 'alter-separate', 'alter-best', 'alter-all', 'gen-a',
            'e2e-lies', 'e2e-highConf', 'e2e-all',
            'dump-placeholder', 'dump-lies', 'dump-highConf', 'dump-hcSeparate', 'dump-altAll']

CONCEPT_NET_FILE = '../data/cn_relations_orig.txt'
CORENLP_CACHES = {
    'dev': '../data/squadv2.0/dev_split_v2_corenlp_cache.json',
    'train': '../data/squadv2.0/train_v2_corenlp_cache.json',
}
NEARBY_GLOVE_FILE = '../data/nearby_n100_glove_6B_100d.json'
POSTAG_FILE = '../data/postag_dict.json'
CANDIDATES_FILE = '../data/squadv2.0/answer_candidates.json'
PERTURB_ANS_FILE = '../data/squadv2.0/squad_perturbations.json'
PERTURB_QUES_FILE = '../data/squadv2.0/squad_syntactic_paraphrases.json'

def load_conceptnet(conceptnet_file, relations=None):

    logger.info("Reading ConceptNet")
    with open(conceptnet_file, 'r', encoding='utf-8') as f:
        kb_all = f.readlines()

    kb = defaultdict(lambda: defaultdict(lambda: []))
    for line in kb_all:
        e1, relation, e2 = [token.strip() for token in line.split(',')]
        if relations is not None and relation not in relations:
            continue
        kb[relation][e1].append(e2)
    return kb

conceptnet_kb = load_conceptnet(CONCEPT_NET_FILE, ['Antonym'])

def is_content_word(pos_tag):
    if pos_tag not in ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS',
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return False
    else:
        return  True



def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_cache():
    logger.info("Reading CoreNLP cache file")
    cache_file = CORENLP_CACHES['train']
    with open(cache_file) as f:
        return json.load(f)


def load_postag_dict():
    with open(POSTAG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_nearby_words():
    with open(NEARBY_GLOVE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_candidates():
    mode = 'train'
    with open(CANDIDATES_FILE.replace('answer', mode + '_answer'), 'r', encoding='utf-8') as f:
        return json.load(f)


def compress_whnp(tree, inside_whnp=False):
    if not tree.children: return tree  # Reached leaf
    # Compress all children
    for i, c in enumerate(tree.children):
        tree.children[i] = compress_whnp(c, inside_whnp=inside_whnp or tree.tag == 'WHNP')
    if tree.tag != 'WHNP':
        if inside_whnp:
            # Wrap everything in an NP
            return corenlp.ConstituencyParse('NP', children=[tree])
        return tree
    wh_word = None
    new_np_children = []
    new_siblings = []
    for i, c in enumerate(tree.children):
        if i == 0:
            if c.tag in ('WHNP', 'WHADJP', 'WHAVP', 'WHPP'):
                wh_word = c.children[0]
                new_np_children.extend(c.children[1:])
            elif c.tag in ('WDT', 'WP', 'WP$', 'WRB'):
                wh_word = c
            else:
                # No WH-word at start of WHNP
                return tree
        else:
            if c.tag == 'SQ':  # Due to bad parse, SQ may show up here
                new_siblings = tree.children[i:]
                break
            # Wrap everything in an NP
            new_np_children.append(corenlp.ConstituencyParse('NP', children=[c]))
    if new_np_children:
        new_np = corenlp.ConstituencyParse('NP', children=new_np_children)
        new_tree = corenlp.ConstituencyParse('WHNP', children=[wh_word, new_np])
    else:
        new_tree = tree
    if new_siblings:
        new_tree = corenlp.ConstituencyParse('SBARQ', children=[new_tree] + new_siblings)
    return new_tree


def read_const_parse(parse_str):
    tree = corenlp.ConstituencyParse.from_corenlp(parse_str)
    new_tree = compress_whnp(tree)
    return new_tree


### Rules for converting questions into declarative sentences
def fix_style(s):
    """Minor, general style fixes for questions."""
    s = s.replace('?', '')  # Delete question marks anywhere in sentence.
    s = s.strip(' .')
    if s[0] == s[0].lower():
        s = s[0].upper() + s[1:]
    return s + '.'


CONST_PARSE_MACROS = {
    '$Noun': '$NP/$NN/$NNS/$NNP/$NNPS',
    '$Verb': '$VB/$VBD/$VBP/$VBZ',
    '$Part': '$VBN/$VG',
    '$Be': 'is/are/was/were',
    '$Do': "do/did/does/don't/didn't/doesn't",
    '$WHP': '$WHADJP/$WHADVP/$WHNP/$WHPP',
}


def _check_match(node, pattern_tok):
    if pattern_tok in CONST_PARSE_MACROS:
        pattern_tok = CONST_PARSE_MACROS[pattern_tok]
    if ':' in pattern_tok:
        # ':' means you match the LHS category and start with something on the right
        lhs, rhs = pattern_tok.split(':')
        match_lhs = _check_match(node, lhs)
        if not match_lhs: return False
        phrase = node.get_phrase().lower()
        retval = any(phrase.startswith(w) for w in rhs.split('/'))
        return retval
    elif '/' in pattern_tok:
        return any(_check_match(node, t) for t in pattern_tok.split('/'))
    return ((pattern_tok.startswith('$') and pattern_tok[1:] == node.tag) or
            (node.word and pattern_tok.lower() == node.word.lower()))


def _recursive_match_pattern(pattern_toks, stack, matches):
    """Recursively try to match a pattern, greedily."""
    if len(matches) == len(pattern_toks):
        # We matched everything in the pattern; also need stack to be empty
        return len(stack) == 0
    if len(stack) == 0: return False
    cur_tok = pattern_toks[len(matches)]
    node = stack.pop()
    # See if we match the current token at this level
    is_match = _check_match(node, cur_tok)
    if is_match:
        cur_num_matches = len(matches)
        matches.append(node)
        new_stack = list(stack)
        success = _recursive_match_pattern(pattern_toks, new_stack, matches)
        if success: return True
        # Backtrack
        while len(matches) > cur_num_matches:
            matches.pop()
    # Recurse to children
    if not node.children:
        return False  # No children to recurse on, we failed
    stack.extend(node.children[::-1])  # Leftmost children should be popped first
    return _recursive_match_pattern(pattern_toks, stack, matches)


def match_pattern(pattern, const_parse):
    pattern_toks = pattern.split(' ')
    whole_phrase = const_parse.get_phrase()
    if whole_phrase.endswith('?') or whole_phrase.endswith('.'):
        # Match trailing punctuation as needed
        pattern_toks.append(whole_phrase[-1])
    matches = []
    success = _recursive_match_pattern(pattern_toks, [const_parse], matches)
    if success:
        return matches
    else:
        return None


def run_postprocessing(s, rules, all_args):
    rule_list = rules.split(',')
    for rule in rule_list:
        if rule == 'lower':
            s = s.lower()
        elif rule.startswith('tense-'):
            ind = int(rule[6:])
            orig_vb = all_args[ind]
            tenses = patten.tenses(orig_vb)
            for tense in PATTERN_TENSES:  # Prioritize by PATTERN_TENSES
                if tense in tenses:
                    break
            else:  # Default to first tense
                tense = PATTERN_TENSES[0]
            s = patten.conjugate(s, tense)
        elif rule in POS_TO_PATTERN:
            s = patten.conjugate(s, POS_TO_PATTERN[rule])
    return s


def convert_whp(node, q, a, tokens):
    if node.tag in ('WHNP', 'WHADJP', 'WHADVP', 'WHPP'):
        # Apply WHP rules
        cur_phrase = node.get_phrase()
        cur_tokens = tokens[node.get_start_index():node.get_end_index()]
        for r in WHP_RULES:
            phrase = r.convert(cur_phrase, a, cur_tokens, node, run_fix_style=False)
            if phrase:
                if not QUIET:
                    print(('  WHP Rule "%s": %s' % (r.name, colored(phrase, 'yellow'))))
                return phrase
    return None


class ConversionRule(object):
    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        raise NotImplementedError


class ConstituencyRule(ConversionRule):
    """A rule for converting question to sentence based on constituency parse."""

    def __init__(self, in_pattern, out_pattern, postproc=None):
        self.in_pattern = in_pattern  # e.g. "where did $NP $VP"
        self.out_pattern = out_pattern
        # e.g. "{1} did {2} at {0}."  Answer is always 0
        self.name = in_pattern
        if postproc:
            self.postproc = postproc
        else:
            self.postproc = {}

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        pattern_toks = self.in_pattern.split(' ')  # Don't care about trailing punctuation
        match = match_pattern(self.in_pattern, const_parse)
        appended_clause = False
        if not match:
            # Try adding a PP at the beginning
            appended_clause = True
            new_pattern = '$PP , ' + self.in_pattern
            pattern_toks = new_pattern.split(' ')
            match = match_pattern(new_pattern, const_parse)
        if not match:
            # Try adding an SBAR at the beginning
            new_pattern = '$SBAR , ' + self.in_pattern
            pattern_toks = new_pattern.split(' ')
            match = match_pattern(new_pattern, const_parse)
        if not match: return None
        appended_clause_match = None
        fmt_args = [a]
        for t, m in zip(pattern_toks, match):
            if t.startswith('$') or '/' in t:
                # First check if it's a WHP
                phrase = convert_whp(m, q, a, tokens)
                if not phrase:
                    phrase = m.get_phrase()
                fmt_args.append(phrase)
        if appended_clause:
            appended_clause_match = fmt_args[1]
            fmt_args = [a] + fmt_args[2:]
        for i in range(len(fmt_args)):
            if i in self.postproc:
                # Run postprocessing filters
                fmt_args[i] = run_postprocessing(fmt_args[i], self.postproc[i], fmt_args)
        output = self.gen_output(fmt_args)
        if appended_clause:
            output = appended_clause_match + ', ' + output
        if run_fix_style:
            output = fix_style(output)
        return output

    def gen_output(self, fmt_args):
        """By default, use self.out_pattern.  Can be overridden."""
        return self.out_pattern.format(*fmt_args)


class ReplaceRule(ConversionRule):
    """A simple rule that replaces some tokens with the answer."""

    def __init__(self, target, replacement='{}', start=False):
        self.target = target
        self.replacement = replacement
        self.name = 'replace(%s)' % target
        self.start = start

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        t_toks = self.target.split(' ')
        q_toks = q.rstrip('?.').split(' ')
        replacement_text = self.replacement.format(a)
        for i in range(len(q_toks)):
            if self.start and i != 0: continue
            if ' '.join(q_toks[i:i + len(t_toks)]).rstrip(',').lower() == self.target:
                begin = q_toks[:i]
                end = q_toks[i + len(t_toks):]
                output = ' '.join(begin + [replacement_text] + end)
                if run_fix_style:
                    output = fix_style(output)
                return output
        return None


class FindWHPRule(ConversionRule):
    """A rule that looks for $WHP's from right to left and does replacements."""
    name = 'FindWHP'

    def _recursive_convert(self, node, q, a, tokens, found_whp):
        if node.word: return node.word, found_whp
        if not found_whp:
            whp_phrase = convert_whp(node, q, a, tokens)
            if whp_phrase: return whp_phrase, True
        child_phrases = []
        for c in node.children[::-1]:
            c_phrase, found_whp = self._recursive_convert(c, q, a, tokens, found_whp)
            child_phrases.append(c_phrase)
        out_toks = []
        for i, p in enumerate(child_phrases[::-1]):
            if i == 0 or p.startswith("'"):
                out_toks.append(p)
            else:
                out_toks.append(' ' + p)
        return ''.join(out_toks), found_whp

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        out_phrase, found_whp = self._recursive_convert(const_parse, q, a, tokens, False)
        if found_whp:
            if run_fix_style:
                out_phrase = fix_style(out_phrase)
            return out_phrase
        return None


class AnswerRule(ConversionRule):
    """Just return the answer."""
    name = 'AnswerRule'

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        return a


CONVERSION_RULES = [
    # Special rules
    ConstituencyRule('$WHP:what $Be $NP called that $VP', '{2} that {3} {1} called {1}'),

    # What type of X
    # ConstituencyRule("$WHP:what/which type/sort/kind/group of $NP/$Noun $Be $NP", '{5} {4} a {1} {3}'),
    # ConstituencyRule("$WHP:what/which type/sort/kind/group of $NP/$Noun $Be $VP", '{1} {3} {4} {5}'),
    # ConstituencyRule("$WHP:what/which type/sort/kind/group of $NP $VP", '{1} {3} {4}'),

    # How $JJ
    ConstituencyRule('how $JJ $Be $NP $IN $NP', '{3} {2} {0} {1} {4} {5}'),
    ConstituencyRule('how $JJ $Be $NP $SBAR', '{3} {2} {0} {1} {4}'),
    ConstituencyRule('how $JJ $Be $NP', '{3} {2} {0} {1}'),

    # When/where $Verb
    ConstituencyRule('$WHP:when/where $Do $NP', '{3} occurred in {1}'),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb', '{3} {4} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP/$PP', '{3} {4} {5} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP $PP', '{3} {4} {5} {6} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Be $NP', '{3} {2} in {1}'),
    ConstituencyRule('$WHP:when/where $Verb $NP $VP/$ADJP', '{3} {2} {4} in {1}'),

    # What/who/how $Do
    ConstituencyRule("$WHP:what/which/who $Do $NP do", '{3} {1}', {0: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb", '{3} {4} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $IN/$NP", '{3} {4} {5} {1}', {4: 'tense-2', 0: 'vbg'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $PP", '{3} {4} {1} {5}', {4: 'tense-2', 0: 'vbg'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $NP $VP", '{3} {4} {5} {6} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB", '{3} {4} to {5} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB $VP", '{3} {4} to {5} {1} {6}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $NP $IN $VP", '{3} {4} {5} {6} {1} {7}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $PP/$S/$VP/$SBAR/$SQ", '{3} {4} {1} {5}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $PP $PP/$S/$VP/$SBAR", '{3} {4} {1} {5} {6}',
                     {4: 'tense-2'}),

    # What/who/how $Be
    # Watch out for things that end in a preposition
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP of $NP $Verb/$Part $IN", '{3} of {4} {2} {5} {6} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $NP $IN", '{3} {2} {4} {5} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $VP/$IN", '{3} {2} {4} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $IN $NP/$VP", '{1} {2} {3} {4} {5}'),
    ConstituencyRule('$WHP:what/which/who $Be/$MD $NP $Verb $PP', '{3} {2} {4} {1} {5}'),
    ConstituencyRule('$WHP:what/which/who $Be/$MD $NP/$VP/$PP', '{1} {2} {3}'),
    ConstituencyRule("$WHP:how $Be/$MD $NP $VP", '{3} {2} {4} by {1}'),

    # What/who $Verb
    ConstituencyRule("$WHP:what/which/who $VP", '{1} {2}'),

    # $IN what/which $NP
    ConstituencyRule('$IN what/which $NP $Do $NP $Verb $NP', '{5} {6} {7} {1} the {3} of {0}',
                     {1: 'lower', 6: 'tense-4'}),
    ConstituencyRule('$IN what/which $NP $Be $NP $VP/$ADJP', '{5} {4} {6} {1} the {3} of {0}',
                     {1: 'lower'}),
    ConstituencyRule('$IN what/which $NP $Verb $NP/$ADJP $VP', '{5} {4} {6} {1} the {3} of {0}',
                     {1: 'lower'}),
    FindWHPRule(),
]

# Rules for going from WHP to an answer constituent
WHP_RULES = [
    # WHPP rules
    ConstituencyRule('$IN what/which type/sort/kind/group of $NP/$Noun', '{1} {0} {4}'),
    ConstituencyRule('$IN what/which type/sort/kind/group of $NP/$Noun $PP', '{1} {0} {4} {5}'),
    ConstituencyRule('$IN what/which $NP', '{1} the {3} of {0}'),
    ConstituencyRule('$IN $WP/$WDT', '{1} {0}'),

    # what/which
    ConstituencyRule('what/which type/sort/kind/group of $NP/$Noun', '{0} {3}'),
    ConstituencyRule('what/which type/sort/kind/group of $NP/$Noun $PP', '{0} {3} {4}'),
    ConstituencyRule('what/which $NP', 'the {2} of {0}'),

    # How many
    ConstituencyRule('how many/much $NP', '{0} {2}'),

    # Replace
    ReplaceRule('what'),
    ReplaceRule('who'),
    ReplaceRule('how many'),
    ReplaceRule('how much'),
    ReplaceRule('which'),
    ReplaceRule('where'),
    ReplaceRule('when'),
    ReplaceRule('why'),
    ReplaceRule('how'),

    # Just give the answer
    AnswerRule(),
]


def get_qas(dataset):
    qas = []
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question'].strip()
                answers = sorted(qa['answers'],
                                 key=lambda x: len(x['text']))  # Prefer shorter answers
                qas.append((question, answers, paragraph['context']))
    return qas


def print_questions(qas):
    qas = sorted(qas, key=lambda x: x[0])
    for question, answers, context in qas:
        print(question.encode('utf-8'))


def print_answers(qas):
    for question, answers, context in qas:
        toks = list(answers)
        toks[0] = colored(answers[0]['text'], 'cyan')
        print(', '.join(toks).encode('utf-8'))


def run_corenlp(dataset, qas):
    cache = {}
    with corenlp.CoreNLPServer(port=CORENLP_PORT, logfile=CORENLP_LOG) as server:
        client = corenlp.CoreNLPClient(port=CORENLP_PORT)
        print('Running NER for paragraphs...', file=sys.stderr)
        for article in dataset['data']:
            for paragraph in article['paragraphs']:
                response = client.query_ner(paragraph['context'])
                cache[paragraph['context']] = response
        print('Parsing questions...', file=sys.stderr)
        for question, answers, context in qas:
            response = client.query_const_parse(question, add_ner=True)
            cache[question] = response['sentences'][0]
    cache_file = CORENLP_CACHES[OPTS.dataset]
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)


def run_conversion(qas):
    corenlp_cache = load_cache()
    rule_counter = collections.Counter()
    unmatched_qas = []
    num_matched = 0
    for question, answers, context in qas:
        parse = corenlp_cache[question]
        tokens = parse['tokens']
        const_parse = read_const_parse(parse['parse'])
        answer = answers[0]['text']
        if not QUIET:
            print(question.encode('utf-8'))
        for rule in CONVERSION_RULES:
            sent = rule.convert(question, answer, tokens, const_parse)
            if sent:
                if not QUIET:
                    print(('  Rule "%s": %s' % (rule.name, colored(sent, 'green'))
                           ).encode('utf-8'))
                rule_counter[rule.name] += 1
                num_matched += 1
                break
        else:
            unmatched_qas.append((question, answer))
    # Print stats
    if not QUIET:
        print('')
    print('=== Summary ===')
    print('Matched %d/%d = %.2f%% questions' % (
        num_matched, len(qas), 100.0 * num_matched / len(qas)))
    for rule in CONVERSION_RULES:
        num = rule_counter[rule.name]
        print('  Rule "%s" used %d times = %.2f%%' % (
            rule.name, num, 100.0 * num / len(qas)))

    print('')
    print('=== Sampled unmatched questions ===')
    for q, a in sorted(random.sample(unmatched_qas, 20), key=lambda x: x[0]):
        print(('%s [%s]' % (q, colored(a, 'cyan'))).encode('utf-8'))
        parse = corenlp_cache[q]
        const_parse = read_const_parse(parse['parse'])
        # const_parse.print_tree()


def inspect_rule(qas, rule_name):
    corenlp_cache = load_cache()
    num_matched = 0
    rule = CONVERSION_RULES[rule_name]
    for question, answers, context in qas:
        parse = corenlp_cache[question]
        answer = answers[0]['text']
        func = rule(question, parse)
        if func:
            sent = colored(func(answer), 'green')
            print(question.encode('utf-8'))
            print(('  Rule "%s": %s' % (rule_name, sent)).encode('utf-8'))
            num_matched += 1
    print('')
    print('Rule "%s" used %d times = %.2f%%' % (
        rule_name, num_matched, 100.0 * num_matched / len(qas)))


##########
# Rules for altering words in a sentence/question/answer
# Takes a CoreNLP token as input
##########
SPECIAL_ALTERATIONS = {
    'States': 'Kingdom',
    'US': 'UK',
    'U.S': 'U.K.',
    'U.S.': 'U.K.',
    'UK': 'US',
    'U.K.': 'U.S.',
    'U.K': 'U.S.',
    'largest': 'smallest',
    'smallest': 'largest',
    'highest': 'lowest',
    'lowest': 'highest',
    'May': 'April',
    'Peyton': 'Trevor',
}

DO_NOT_ALTER = ['many', 'such', 'few', 'much', 'other', 'same', 'general',
                'type', 'record', 'kind', 'sort', 'part', 'form', 'terms', 'use',
                'place', 'way', 'old', 'young', 'bowl', 'united', 'one',
                'likely', 'different', 'square', 'war', 'republic', 'doctor', 'color']
BAD_ALTERATIONS = ['mx2004', 'planet', 'u.s.', 'Http://Www.Co.Mo.Md.Us']


def alter_special(token, **kwargs):
    w = token['originalText']
    if w in SPECIAL_ALTERATIONS:
        return [SPECIAL_ALTERATIONS[w]]
    return None


def alter_nearby(pos_list, ignore_pos=False, is_ner=False):
    def func(token, nearby_word_dict=None, postag_dict=None, **kwargs):
        if token['pos'] not in pos_list: return None
        if is_ner and token['ner'] not in ('PERSON', 'LOCATION', 'ORGANIZATION', 'MISC'):
            return None
        w = token['word'].lower()
        if w in ('war'): return None
        if w not in nearby_word_dict: return None
        new_words = []
        w_stem = STEMMER.stem(w.replace('.', ''))
        for x in nearby_word_dict[w][1:]:
            new_word = x['word']
            # Make sure words aren't too similar (e.g. same stem)
            new_stem = STEMMER.stem(new_word.replace('.', ''))
            if w_stem.startswith(new_stem) or new_stem.startswith(w_stem): continue
            if not ignore_pos:
                # Check for POS tag match
                if new_word not in postag_dict: continue
                new_postag = postag_dict[new_word]
                if new_postag != token['pos']: continue
            new_words.append(new_word)
        return new_words

    return func


def alter_entity_glove(token, nearby_word_dict=None, **kwargs):
    # NOTE: Deprecated
    if token['ner'] not in ('PERSON', 'LOCATION', 'ORGANIZATION', 'MISC'): return None
    w = token['word'].lower()
    if w == token['word']: return None  # Only do capitalized words
    if w not in nearby_word_dict: return None
    new_words = []
    for x in nearby_word_dict[w][1:3]:
        if token['word'] == w.upper():
            new_words.append(x['word'].upper())
        else:
            new_words.append(x['word'].title())
    return new_words


def alter_entity_type(token, **kwargs):
    pos = token['pos']
    ner = token['ner']
    word = token['word']
    is_abbrev = word == word.upper() and not word == word.lower()
    if token['pos'] not in (
            'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS',
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):
        # Don't alter non-content words
        return None
    if ner == 'PERSON':
        return random.choices(answer_candidates['PERSON'], k=5)
    elif ner == 'LOCATION':
        return random.choices(answer_candidates['LOCATION'], k=5)
    elif ner == 'ORGANIZATION':
        if is_abbrev: return random.choices(['UNICEF', 'UPSC', 'HRD', 'DoD', 'FBI'], k=2)
        return random.choices(answer_candidates['ORGANIZATION'], k=5)
    elif ner == 'MISC':
        return random.choices(answer_candidates['MISC'], k=5)
    elif pos == 'NNP':
        if is_abbrev: return ['XKCD']
        return random.choices(answer_candidates['NNP'], k=5)
    elif pos == 'NNPS':
        return random.choices(answer_candidates['NNPS'], k=5)
    return None


def alter_wordnet_antonyms(token, **kwargs):
    if token['pos'] not in POS_TO_WORDNET: return None
    w = token['word'].lower()
    wn_pos = POS_TO_WORDNET[token['pos']]
    synsets = wn.synsets(w, wn_pos)
    if not synsets: return None
    synset = synsets[0]
    antonyms = []
    for lem in synset.lemmas():
        if lem.antonyms():
            for a in lem.antonyms():
                new_word = a.name()
                if '_' in a.name(): continue
                antonyms.append(new_word)
    return antonyms


def alter_conceptnet_antonyms(token, **kwargs):

    if not is_content_word(token['pos']):
        return None

    word = token['word']
    antonyms = None
    if word in conceptnet_kb['Antonym'].keys():
        antonyms = conceptnet_kb['Antonym'][word]
    elif word.lower() in conceptnet_kb['Antonym'].keys():
        antonyms = conceptnet_kb['Antonym'][word.lower()]
    elif word.capitalize() in conceptnet_kb['Antonym'].keys():
        antonyms = conceptnet_kb['Antonym'][word.capitalize()]
    else:
        return None

    filtered_antonyms = []
    for antonym in antonyms:
        if '_' not in antonym:
            filtered_antonyms.append(antonym)

    return filtered_antonyms

def alter_wordnet_hypernyms(token, **kwargs):
    if token['pos'] not in POS_TO_WORDNET: return None
    w = token['word'].lower()
    wn_pos = POS_TO_WORDNET[token['pos']]
    synsets = wn.synsets(w, wn_pos)
    if not synsets: return None
    all_hypernyms = []
    for synset in synsets:
        hypernyms = synset.hypernyms()
        for word in hypernyms:
            new_word = word.name()
            if '_' in new_word:
                continue
            all_hypernyms.append(new_word)

    return all_hypernyms


HIGH_CONF_ALTER_RULES = collections.OrderedDict([
    ('special', alter_special),
    ('wn_antonyms', alter_wordnet_antonyms),
    ('cn_antonyms', alter_conceptnet_antonyms),
    ('nearbyNum', alter_nearby(['CD'], ignore_pos=True)),
    ('nearbyProperNoun', alter_nearby(['NNP', 'NNPS'])),
    ('nearbyProperNounPos', alter_nearby(['NNP', 'NNPS'], ignore_pos=True)),
    ('nearbyEntityNouns', alter_nearby(['NN', 'NNS'], is_ner=True)),
    ('nearbyEntityJJ', alter_nearby(['JJ', 'JJR', 'JJS'], is_ner=True)),
    ('entityType', alter_entity_type),
    # ('entity_glove', alter_entity_glove),
])
ALL_ALTER_RULES = collections.OrderedDict(list(HIGH_CONF_ALTER_RULES.items()) + [
    ('nearbyAdj', alter_nearby(['JJ', 'JJR', 'JJS'])),
    ('nearbyNoun', alter_nearby(['NN', 'NNS'])),
    # ('nearbyNoun', alter_nearby(['NN', 'NNS'], ignore_pos=True)),
])

def get_unknown_answer(question):
    " Randomly sample an answer candidate for questions that don't have an answer"
    question = question.lower()
    answer = None
    if question.startswith('where'):
        answer = random.choice(answer_candidates['LOCATION'])
    if question.startswith('when'):
        answer = random.choice(answer_candidates['DATE'])
    if question.startswith('who'):
        answer = random.choice(answer_candidates['PERSON'])
    if question.startswith('which'):
        answer = random.choice(answer_candidates['ORGANIZATION'])
    if not answer:
        key = random.choice(list(answer_candidates.keys()))
        answer = random.choice(answer_candidates[key])
    return answer

def alter_answer(answer_corenlp_obj, ans_token_idxs, nearby_word_dict, postag_dict,
                   strategy='high-conf'):
    """Alter the question to make it ask something else.

    Possible strategies:
    - high-conf: Do all possible high-confidence alterations
    - all: Do all possible alterations (very conservative)
    """

    answer_tokens = [t for t in answer_corenlp_obj['tokens']]
    answer = corenlp.rejoin(answer_tokens)
    used_words = [t['word'].lower() for t in answer_tokens]
    new_answers = []
    used_rules = []
    toks_all = []
    if strategy.startswith('high-conf'):
        rules = HIGH_CONF_ALTER_RULES
    else:
        rules = ALL_ALTER_RULES
    for i, t in enumerate(answer_tokens):
        if t['word'].lower() in DO_NOT_ALTER:
            if strategy in ('high-conf', 'all'):
                toks_all.append(t)
            continue
        if i in ans_token_idxs:
            toks_all.append(t)
            continue
        found = False
        for rule_name in rules:
            rule = rules[rule_name]
            new_words = rule(t, nearby_word_dict=nearby_word_dict,
                             postag_dict=postag_dict)
            if new_words:
                # random.shuffle(new_words)
                for nw in new_words:
                    if nw.lower() in used_words: continue
                    if nw.lower() in BAD_ALTERATIONS: continue
                    # Match capitzliation
                    if t['word'] == t['word'].upper():
                        nw = nw.upper()
                    elif t['word'] == t['word'].title():
                        nw = nw.title()
                    new_tok = dict(t)
                    new_tok['word'] = new_tok['lemma'] = new_tok['originalText'] = nw
                    new_tok['altered'] = True
                    # NOTE: obviously this is approximate
                    if strategy in ('high-conf', 'all'):
                        toks_all.append(new_tok)
                        used_rules.append(rule_name)
                        found = True
                        break
                    else:
                        raise NotImplementedError('Not implemented for % strategy' % strategy)
            if strategy in ('high-conf', 'all') and found:
                break
        if strategy in ('high-conf', 'all') and not found:
            toks_all.append(t)
    if strategy in ('high-conf', 'all'):
        new_answer = corenlp.rejoin(toks_all)
        if new_answer != answer:
            new_answers.append((corenlp.rejoin(toks_all), toks_all, strategy))

    return new_answers

def alter_question(q, tokens, const_parse, nearby_word_dict, postag_dict,
                   strategy='separate', k=1):
    """Alter the question to make it ask something else.

  Possible strategies:
    - separate: Do best alteration for each word separately.
    - best: Generate exactly one best alteration (may over-alter).
    - high-conf: Do all possible high-confidence alterations
    - high-conf-separate: Do best high-confidence alteration for each word separately.
    - all: Do all possible alterations (very conservative)
  """
    used_words = [t['word'].lower() for t in tokens]
    new_qs = []

    for _ in range(k):
        toks_all = []
        if strategy.startswith('high-conf'):
            rules = HIGH_CONF_ALTER_RULES
        else:
            rules = ALL_ALTER_RULES
        for i, t in enumerate(tokens):
            if t['word'].lower() in DO_NOT_ALTER:
                if strategy in ('high-conf', 'all'): toks_all.append(t)
                continue
            begin = tokens[:i]
            end = tokens[i + 1:]
            found = False
            for rule_name in rules:
                rule = rules[rule_name]
                new_words = rule(t, nearby_word_dict=nearby_word_dict,
                                 postag_dict=postag_dict)
                if new_words:
                    random.shuffle(new_words)
                    for nw in new_words:
                        if nw.lower() in used_words: continue
                        if nw.lower() in BAD_ALTERATIONS: continue
                        # Match capitzliation
                        if t['word'] == t['word'].upper():
                            nw = nw.upper()
                        elif t['word'] == t['word'].title():
                            nw = nw.title()
                        new_tok = dict(t)
                        new_tok['word'] = new_tok['lemma'] = new_tok['originalText'] = nw
                        used_words.append(nw)
                        new_tok['altered'] = True
                        # NOTE: obviously this is approximate
                        if strategy.endswith('separate'):
                            new_tokens = begin + [new_tok] + end
                            new_q = corenlp.rejoin(new_tokens)
                            tag = '%s-%d-%s' % (rule_name, i, nw)
                            new_const_parse = corenlp.ConstituencyParse.replace_words(
                                const_parse, [t['word'] for t in new_tokens])
                            new_qs.append((new_q, new_tokens, new_const_parse, tag))
                            break
                        elif strategy in ('high-conf', 'all'):
                            toks_all.append(new_tok)
                            found = True
                            break
                if strategy in ('high-conf', 'all') and found: break
            if strategy in ('high-conf', 'all') and not found:
                toks_all.append(t)
        if strategy in ('high-conf', 'all'):
            new_q = corenlp.rejoin(toks_all)
            new_const_parse = corenlp.ConstituencyParse.replace_words(
                const_parse, [t['word'] for t in toks_all])
            if new_q != q:
                new_qs.append((corenlp.rejoin(toks_all), toks_all, new_const_parse, strategy))

    return new_qs


def colorize_alterations(tokens):
    out_toks = []
    for t in tokens:
        if 'altered' in t:
            new_tok = {'originalText': colored(t['originalText'], 'cyan'),
                       'before': t['before']}
            out_toks.append(new_tok)
        else:
            out_toks.append(t)
    return corenlp.rejoin(out_toks)

POLICY_POS_MAP = {0: 'Noun', 1: 'Verb', 2: 'Adverb', 3:'Adjective'}
POLICY_PROB_MAP = {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
POLICY_NER_MAP = {0: 'PERSON', 1: 'LOCATION', 2: 'ORGANIZATION', 3:'MISC', 4:'TEMPORAL', 5:'NUMERIC'}
POLICY_LOC_MAP = {0: 0.25, 1: 0.50, 2: 0.75, 3: 1.0}
POS_TYPES = {'Noun': ['NN', 'NNS', 'NNP', 'NNS'],
             'Verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
             'Adverb': ['RB', 'RBR', 'RBS'],
             'Adjective': ['JJ', 'JJR', 'JJS']}

def find_candidates(tokens, type='pos'):
    candidates = []
    if type == 'pos':
        for key, val in POLICY_POS_MAP.items():
            if any([t['pos'] in POS_TYPES[POLICY_POS_MAP[key]] for t in tokens]):
                candidates.append(key)
    if type == 'ner':
        for key, val in POLICY_NER_MAP.items():
            if any([t['ner'] == val for t in tokens]):
                candidates.append(key)

    return candidates


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
        if corenlp.rejoin(a_toks).strip() == a_obj['text']:
            # Make sure that the tokens reconstruct the answer
            return i, a_toks, a_idx
        if i == 0: first_a_toks = a_toks
    # None of the extracted token lists reconstruct the answer
    # Default to the first
    return 0, first_a_toks, a_idx


def get_determiner_for_answers(answer_objs):
    for a in answer_objs:
        words = a['text'].split(' ')
        if words[0].lower() == 'the': return 'the'
        if words[0].lower() in ('a', 'an'): return 'a'
    return None


def ans_number(a, tokens, q, **kwargs):
    out_toks = []
    seen_num = False
    for t in tokens:
        ner = t['ner']
        pos = t['pos']
        w = t['word']
        out_tok = {'before': t['before']}

        # Split on dashes
        leftover = ''
        dash_toks = w.split('-')
        if len(dash_toks) > 1:
            w = dash_toks[0]
            leftover = '-'.join(dash_toks[1:])

        # Try to get a number out
        value = None
        if w != '%':
            # Percent sign should just pass through
            try:
                value = float(w.replace(',', ''))
            except:
                try:
                    norm_ner = t['normalizedNER']
                    if norm_ner[0] in ('%', '>', '<'):
                        norm_ner = norm_ner[1:]
                    value = float(norm_ner)
                except:
                    pass
        if not value and (
                ner == 'NUMBER' or
                (ner == 'PERCENT' and pos == 'CD')):
            # Force this to be a number anyways
            value = 10
        if value:
            if math.isinf(value) or math.isnan(value): value = random.choice(range(0, 10000))
            seen_num = True
            options = ['thousand', 'million', 'billion', 'trillion']
            if w in options:
                new_val = random.choice([opt for opt in options if opt != w])
            else:
                if value < 2500 and value > 1000:
                    new_val = str(value - random.choice(range(0, 900)))
                else:
                    # Change leading digit
                    if value == int(value):
                        val_chars = list('%d' % value)
                    else:
                        val_chars = list('%g' % value)
                    c = val_chars[0]
                    for i in range(len(val_chars)):
                        c = val_chars[i]
                        if c >= '0' and c <= '9':
                            val_chars[i] = str(random.choice([i for i in range(0, 10) if i != int(c)]))
                            break
                    new_val = ''.join(val_chars)
            if leftover:
                new_val = '%s-%s' % (new_val, leftover)
            out_tok['originalText'] = new_val
        else:
            out_tok['originalText'] = t['originalText']
        out_toks.append(out_tok)
    if seen_num:
        return corenlp.rejoin(out_toks).strip()
    else:
        return None


MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
          'august', 'september', 'october', 'november', 'december']


def ans_date(a, tokens, q, **kwargs):
    out_toks = []
    if not all(t['ner'] == 'DATE' for t in tokens): return None
    for t in tokens:
        if t['pos'] == 'CD' or t['word'].isdigit():
            try:
                value = int(t['word'])
            except:
                value = 10  # fallback
            if value > 50:
                new_val = str(value - random.choice(range(-2000, 25)))  # Year
            else:  # Day of month
                if value > 15:
                    new_val = str(value - random.choice(range(0, 15)))
                else:
                    new_val = str(value + random.choice(range(0, 16)))
        else:
            if t['word'].lower() in MONTHS:
                m_ind = MONTHS.index(t['word'].lower())
                m_new_ind = random.choice(range(len(MONTHS)))
                new_val = MONTHS[m_new_ind].title() if m_new_ind != m_ind else MONTHS[(m_ind + 6) % 12].title()
            else:
                # Give up
                new_val = t['originalText']
        out_toks.append({'before': t['before'], 'originalText': new_val})
    new_ans = corenlp.rejoin(out_toks).strip()
    if new_ans == a['text']: return None
    return new_ans


def ans_entity_full(ner_tag, candidates):
    """Returns a function that yields new_ans iff every token has |ner_tag|."""

    def func(a, tokens, q, **kwargs):
        for t in tokens:
            if t['ner'] != ner_tag: return None
        return random.choice(candidates)

    return func


def ans_abbrev(candidates):
    def func(a, tokens, q, **kwargs):
        s = a['text']
        if s == s.upper() and s != s.lower():
            return random.choice(candidates)
        return None

    return func


def ans_match_wh(wh_word, candidates):
    """Returns a function that yields new_ans if the question starts with |wh_word|."""

    def func(a, tokens, q, **kwargs):
        if q.lower().startswith(wh_word + ' '):
            return random.choice(candidates)
        return None

    return func


def ans_pos(pos, candidates, end=False, add_dt=False):
    """Returns a function that yields new_ans if the first/last token has |pos|."""

    def func(a, tokens, q, determiner, **kwargs):
        if end:
            t = tokens[-1]
        else:
            t = tokens[0]
        if t['pos'] != pos: return None
        if add_dt and determiner:
            return '%s %s' % (determiner, random.choice(candidates))
        return random.choice(candidates)

    return func


def ans_catch_all(candidates):
    def func(a, tokens, q, **kwargs):
        return random.choice(candidates)

    return func

answer_candidates = load_candidates()

ANSWER_RULES = [
    ('date', ans_date),
    ('number', ans_number),
    ('ner_person', ans_entity_full('PERSON', answer_candidates['PERSON'])),
    ('ner_location', ans_entity_full('LOCATION', answer_candidates['LOCATION'])),
    ('ner_organization', ans_entity_full('ORGANIZATION', answer_candidates['ORGANIZATION'])),
    ('ner_misc', ans_entity_full('MISC', answer_candidates['MISC'])),
    ('abbrev', ans_abbrev(['LSTM', 'UNICEF', 'MMA', 'MVP', 'USPS', 'NOAA'])),
    ('wh_who', ans_match_wh('who', answer_candidates['PERSON'])),
    ('wh_when', ans_match_wh('when', answer_candidates['DATE'])),
    ('wh_where', ans_match_wh('where', answer_candidates['LOCATION'])),
    ('wh_where', ans_match_wh('how many', range(0, 1000))),

    # Starts with verb
    ('pos_begin_vb', ans_pos('VB', answer_candidates['VB'])),
    ('pos_end_vbd', ans_pos('VBD', answer_candidates['VBD'])),
    ('pos_end_vbg', ans_pos('VBG', answer_candidates['VBG'])),
    ('pos_end_vbp', ans_pos('VBP', answer_candidates['VBP'])),
    ('pos_end_vbz', ans_pos('VBZ', answer_candidates['VBZ'])),

    # Ends with some POS tag
    ('pos_end_nn', ans_pos('NN', answer_candidates['NN'], end=True, add_dt=True)),
    ('pos_end_nnp', ans_pos('NNP', answer_candidates['NNP'], end=True, add_dt=True)),
    ('pos_end_nns', ans_pos('NNS', answer_candidates['NNS'], end=True, add_dt=True)),
    ('pos_end_nnps', ans_pos('NNPS', answer_candidates['NNPS'], end=True, add_dt=True)),
    ('pos_end_jj', ans_pos('JJ', answer_candidates['JJ'], end=True)),
    ('pos_end_jjr', ans_pos('JJR', answer_candidates['JJR'], end=True)),
    ('pos_end_jjs', ans_pos('JJS', answer_candidates['JJS'], end=True)),
    ('pos_end_rb', ans_pos('RB', answer_candidates['RB'], end=True)),
    ('pos_end_vbg', ans_pos('VBG', answer_candidates['VBG'], end=True)),

    ('catch_all', ans_catch_all(['aliens', 'tranjectories', 'explorations', 'gamma rays', 'nascent oxygen'])),
]

MOD_ANSWER_RULES = [
    ('date', ans_date),
    ('number', ans_number),
    ('ner_person', ans_entity_full('PERSON', 'Charles Babbage')),
    ('ner_location', ans_entity_full('LOCATION', 'Stockholm')),
    ('ner_organization', ans_entity_full('ORGANIZATION', 'Acme Corporation')),
    ('ner_misc', ans_entity_full('MISC', 'Soylent')),
    ('abbrev', ans_abbrev('PCFG')),
    ('wh_who', ans_match_wh('who', 'Charles Babbage')),
    ('wh_when', ans_match_wh('when', '2004')),
    ('wh_where', ans_match_wh('where', 'Stockholm')),
    ('wh_where', ans_match_wh('how many', '200')),
    # Starts with verb
    ('pos_begin_vb', ans_pos('VB', 'run')),
    ('pos_end_vbd', ans_pos('VBD', 'ran')),
    ('pos_end_vbg', ans_pos('VBG', 'running')),
    ('pos_end_vbp', ans_pos('VBP', 'runs')),
    ('pos_end_vbz', ans_pos('VBZ', 'runs')),
    # Ends with some POS tag
    ('pos_end_nn', ans_pos('NN', 'apple', end=True, add_dt=True)),
    ('pos_end_nnp', ans_pos('NNP', 'Sears Tower', end=True, add_dt=True)),
    ('pos_end_nns', ans_pos('NNS', 'apples', end=True, add_dt=True)),
    ('pos_end_nnps', ans_pos('NNPS', 'Hobbits', end=True, add_dt=True)),
    ('pos_end_jj', ans_pos('JJ', 'blue', end=True)),
    ('pos_end_jjr', ans_pos('JJR', 'bluer', end=True)),
    ('pos_end_jjs', ans_pos('JJS', 'bluest', end=True)),
    ('pos_end_rb', ans_pos('RB', 'quickly', end=True)),
    ('pos_end_vbg', ans_pos('VBG', 'running', end=True)),
    ('catch_all', ans_catch_all('cosmic rays')),
]


def generate_answers(qas):
    corenlp_cache = load_cache()
    # nearby_word_dict = load_nearby_words()
    # postag_dict = load_postag_dict()
    rule_counter = collections.Counter()
    unmatched_qas = []
    num_matched = 0
    for question, answers, context in qas:
        parse = corenlp_cache[context]
        ind, tokens = get_tokens_for_answers(answers, parse)
        determiner = get_determiner_for_answers(answers)
        answer = answers[ind]
        if not QUIET:
            print(('%s [%s]' % (question, colored(answer['text'], 'cyan'))).encode('utf-8'))
        for rule_name, func in ANSWER_RULES:
            new_ans = func(answer, tokens, question, determiner=determiner)
            if new_ans:
                num_matched += 1
                rule_counter[rule_name] += 1
                if not QUIET:
                    print(('  Rule %s: %s' % (rule_name, colored(new_ans, 'green'))).encode('utf-8'))
                break
        else:
            unmatched_qas.append((question, answer['text']))
    # Print stats
    if not QUIET:
        print('')
    print('=== Summary ===')
    print('Matched %d/%d = %.2f%% questions' % (
        num_matched, len(qas), 100.0 * num_matched / len(qas)))
    print('')
    for rule_name, func in ANSWER_RULES:
        num = rule_counter[rule_name]
        print('  Rule "%s" used %d times = %.2f%%' % (
            rule_name, num, 100.0 * num / len(qas)))
    print('')
    print('=== Sampled unmatched answers ===')
    for q, a in sorted(random.sample(unmatched_qas, min(20, len(unmatched_qas))),
                       key=lambda x: x[0]):
        print(('%s [%s]' % (q, colored(a, 'cyan'))).encode('utf-8'))


def dump_answer_candidates(dataset, prefix='train'):
    corenlp_cache = load_cache()
    ner_types = ['PERSON', 'LOCATION', 'ORGANIZATION', 'MISC', 'DATE']
    pos_types = ['VB', 'VBD', 'VBG', 'VBP', 'VBZ', 'NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'VBG']
    answer_candidates = collections.defaultdict(lambda: [])
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                p_parse = corenlp_cache[paragraph['context']]
                ind, a_toks = get_tokens_for_answers(qa['answers'], p_parse)
                answer_obj = qa['answers'][ind]
                for ner_type in ner_types:
                    if all([tok['ner'] == ner_type for tok in a_toks]):
                        answer_candidates[ner_type].append(answer_obj['text'])
                        continue
                for pos_type in pos_types:
                    for token in a_toks:
                        if token['pos'] == pos_type:
                            answer_candidates[pos_type].append(token['word'].lower())
                            continue
    for key, val in answer_candidates.items():
        answer_candidates[key] = list(set(val))
        print(key, list(set(val)))

    with open(os.path.join('out', prefix + '_answer_candidates.json'), 'w') as f:
        json.dump(answer_candidates, f)


def insert_sentences(context, answer_obj, altered_sents, sentence_spans):

    num_sentences = len(sentence_spans)
    insert_locs = random.choices(range(num_sentences + 1), k=len(altered_sents))
    insert_locs.sort()

    insert_spans = []
    new_answers = [{'text': a['text'], 'answer_start': a['answer_start']} for a in answer_obj]

    if insert_locs[0] == 0:
        cur_text = altered_sents[0] + ' '
        for a_idx, a in enumerate(new_answers):
            new_answers[a_idx] = {
                'text': a['text'],
                'answer_start': a['answer_start'] + len(altered_sents[0]) + 1
            }
        insert_spans.append([0, len(altered_sents[0])-1])
        altered_sents = altered_sents[1:]
        insert_locs = insert_locs[1:]
    else:
        cur_text = ''

    pre_insert_sentences = [sentence_spans[insert_loc-1] for insert_loc in insert_locs]
    end_offsets = [sentence[-1] + 1 for sentence in pre_insert_sentences]
    prev_offset = 0

    for end_offset, altered_sent in zip(end_offsets, altered_sents):
        cur_text += context[prev_offset:end_offset] + altered_sent + ' '
        insert_spans.append([end_offset, end_offset + len(altered_sent) - 1])
        for a_idx, a in enumerate(new_answers):
            new_answer_start = a['answer_start'] if a['answer_start'] < end_offset else a['answer_start'] + len(altered_sent) + 1
            new_answers[a_idx] = {
                'text': a['text'],
                'answer_start': new_answer_start
            }
        prev_offset = end_offset
    cur_text = cur_text + context[end_offset:]

    for a in new_answers:
        try:
            assert cur_text[a['answer_start']:(a['answer_start'] + len(a['text']))] == a['text']
        except AssertionError as e:
            print(e)
            print(cur_text[a['answer_start']: (a['answer_start'] + len(a['text']))], a['text'])
            return None, None, None

    return cur_text, new_answers, insert_spans


def insert_sentence(context, answer_obj, altered_sent, sentence_spans):

    num_sentences = len(sentence_spans)
    insert_loc = random.choice(range(num_sentences + 1))

    if insert_loc == 0:
        cur_text = '%s %s' % (altered_sent, context)
        new_answers = []
        for a in answer_obj:
            new_answers.append({
                'text': a['text'],
                'answer_start': a['answer_start'] + len(altered_sent) + 1
            })
        insert_span = (0, len(altered_sent)-1)
    elif insert_loc == num_sentences:
        cur_text = '%s %s' % (context, altered_sent)
        new_answers = answer_obj
        insert_span = (len(context) + 1, len(context) + 1 + len(altered_sent) - 1)
    else:
        end_offset = sentence_spans[insert_loc-1][-1]
        cur_text = context[:end_offset+1] + altered_sent + ' ' + context[end_offset:]
        insert_span = (end_offset + 1, end_offset + 1 + len(altered_sent) - 1)
        new_answers = []
        for a in answer_obj:
            new_answer_start = a['answer_start'] if a['answer_start'] < end_offset else a['answer_start'] + len(altered_sent) + 1
            new_answers.append({
                'text': a['text'],
                'answer_start': new_answer_start
            })
            try:
                assert cur_text[new_answer_start:(new_answer_start + len(a['text']))] == a['text']
            except AssertionError as e:
                print(e)
                print(cur_text[new_answer_start: (new_answer_start + len(a['text']))], a['text'])
                return None, None, None

    return cur_text, new_answers, insert_span

def semantic_diff_check(original_str, altered_str, content_phrases, thresh=1):
    overlaps = 0
    altered = 0
    for phrase in content_phrases:
        phrase_str = ' '.join([t['word'] for i, t in phrase])
        if phrase_str in original_str:
            overlaps += 1
            if phrase_str not in altered_str:
                altered += 1

    if overlaps >= thresh and altered >= thresh:
        return True
    else:
        return False

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
                 'Verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                 'Adverb': ['RB', 'RBR', 'RBS'],
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


class AdversarialAttack:

    def __init__(self, corenlp_cache, nearby_word_dict, postag_dict):

        self.corenlp_cache = corenlp_cache
        self.nearby_word_dict = nearby_word_dict
        self.postag_dict = postag_dict
        self.adversary_to_func = {
            'AddSentDiverse': self.add_sent_diverse,
            'AddKSentDiverse': self.add_k_sent_diverse,
            'InvalidateAnswer': self.invalidate_answer,
            'AddAnswerPosition': self.add_answer_position
        }
        self.version2 = True

    def generate_adversarial_sample(self, qa_sample, context, policy='AddSentDiverse', paraphrase=None):
        adversary_func = self.adversary_to_func[policy]

        if policy == 'AddKSentDiverse':
            return adversary_func(qa_sample, context, paraphrase=paraphrase, num_distractors=2)
        else:
            return adversary_func(qa_sample, context, paraphrase=paraphrase)

    def add_answer_position(self, qa_sample, context, paraphrase=None):

        altered = False
        question = qa_sample['question'].strip()
        print(('Question: %s' % question))

        if qa_sample['answers'] == []:
            return None

        context_parse = self.corenlp_cache[context]
        ind, a_toks, ans_loc = get_tokens_for_answers(qa_sample['answers'], context_parse)
        answer_obj = qa_sample['answers'][ind]
        answer = answer_obj['text']

        ques_parse = self.corenlp_cache[question]
        ques_tokens = ques_parse['tokens']
        ques_content_phrases = extract_phrases(ques_tokens)

        ques_const_parse = read_const_parse(ques_parse['parse'])
        q_list = alter_question(question, ques_tokens, ques_const_parse, self.nearby_word_dict,
                                self.postag_dict, strategy='high-conf')

        sent_counter = 0
        altered_qa_samples = []
        for q_str, q_tokens, q_const_parse, tag in q_list:
            for rule in CONVERSION_RULES:
                sent = rule.convert(q_str, answer, q_tokens, q_const_parse)
                if sent:
                    sent_counter += 1
                    print(('  Choice %d: Distractor Sentence (%s): %s' % (sent_counter, tag, colored(sent, 'cyan'))))
                    if semantic_diff_check(corenlp.rejoin(ques_tokens), sent, ques_content_phrases):
                        pass
                    else:
                        print(colored("Not semantically different enough:", 'red'))
                        continue

                    altered_qa_sample = {
                        'question': qa_sample['question'],
                        'id': '%s-%s' % (qa_sample['id'], tag),
                        'answers': qa_sample['answers']
                    }
                    if self.version2:
                        altered_qa_sample['is_impossible'] = False

                    sentence_spans = [[s['tokens'][0]['characterOffsetBegin'], s['tokens'][-1]['characterOffsetEnd']]
                                      for s in context_parse['sentences']]
                    if paraphrase:
                        cur_text, answers, sentence_spans = self.replace_with_paraphrase(qa_sample, context,
                                                                                         paraphrase['distractor'], sentence_spans, ans_loc)
                    else:
                        cur_text = context
                        answers = qa_sample['answers']

                    cur_text, new_answers, insert_span = insert_sentence(cur_text, answers, sent, sentence_spans)

                    altered_qa_sample['answers'] = new_answers
                    altered_qa_sample['adversarial_span'] = insert_span

                    if not cur_text or not new_answers:
                        continue

                    altered_paragraph = {'context': cur_text, 'qas': [altered_qa_sample]}
                    altered_qa_samples.append(altered_paragraph)
                    altered = True
                    break

        if altered:
            return altered_qa_samples
        else:
            return None

    def add_k_sent_diverse(self, qa_sample, context, paraphrase=None, num_distractors=2):

        question = qa_sample['question'].strip()
        print(('Question: %s' % question))

        context_parse = self.corenlp_cache[context]
        if qa_sample['answers'] != []:
            ind, a_toks, ans_loc = get_tokens_for_answers(qa_sample['answers'], context_parse)
            determiner = get_determiner_for_answers(qa_sample['answers'])
            answer_obj = qa_sample['answers'][ind]
            for rule_name, func in ANSWER_RULES:
                answer = func(answer_obj, a_toks, question, determiner=determiner)
                if answer:
                    answer = str(answer)
                    break
            else:
                raise ValueError('Missing answer')
        else:
            answer = get_unknown_answer(question)

        ques_parse = self.corenlp_cache[question]
        ques_tokens = ques_parse['tokens']
        ques_content_phrases = extract_phrases(ques_tokens)
        ques_const_parse = read_const_parse(ques_parse['parse'])
        q_list = alter_question(
            question, ques_tokens, ques_const_parse, self.nearby_word_dict,
            self.postag_dict, strategy='high-conf', k=num_distractors)

        try:
            assert len(q_list) >= num_distractors
        except:
            print("Could not find %s distractors for this example" % num_distractors)
            return None

        sent_counter = 0
        altered_qa_samples = []
        altered_sentences = []
        for q_str, q_tokens, q_const_parse, tag in q_list:
            for rule in CONVERSION_RULES:
                sent = rule.convert(q_str, answer, q_tokens, q_const_parse)
                if sent:
                    sent_counter += 1
                    print(('  Choice %d: Distractor Sentence (%s): %s' % (sent_counter, tag, colored(sent, 'cyan'))))
                    if not semantic_diff_check(corenlp.rejoin(ques_tokens), sent, ques_content_phrases, thresh=1):
                        print(colored("Not semantically different enough:", 'red'))
                        continue
                    else:
                        altered_sentences.append(sent)
                        break

        if not altered_sentences or len(altered_sentences) < num_distractors:
            return None

        altered_qa_sample = {
            'question': qa_sample['question'],
            'id': '%s-%s' % (qa_sample['id'], tag),
            'answers': qa_sample['answers']
        }

        if self.version2:
            altered_qa_sample['is_impossible'] = False

        sentence_spans = [[s['tokens'][0]['characterOffsetBegin'], s['tokens'][-1]['characterOffsetEnd']]
                          for s in context_parse['sentences']]
        if paraphrase:
            cur_text, answers, sentence_spans = self.replace_with_paraphrase(qa_sample, context, paraphrase['distractor'],
                                                                             sentence_spans, ans_loc)
        else:
            cur_text = context
            answers = qa_sample['answers']

        cur_text, new_answers, insert_span = insert_sentences(cur_text, answers,
                                                              random.choices(altered_sentences, k=num_distractors), sentence_spans)
        altered_qa_sample['answers'] = new_answers
        altered_qa_sample['adversarial_span'] = insert_span

        if not cur_text or not new_answers:
            return None

        altered_paragraph = {'context': cur_text, 'qas': [altered_qa_sample]}
        altered_qa_samples.append(altered_paragraph)
        return altered_qa_samples


    def add_sent_diverse(self, qa_sample, context, paraphrase=None):

        altered = False
        question = qa_sample['question'].strip()
        print(('Question: %s' % question))

        context_parse = self.corenlp_cache[context]
        if qa_sample['answers'] != []:
            ind, a_toks, ans_loc = get_tokens_for_answers(qa_sample['answers'], context_parse)
            determiner = get_determiner_for_answers(qa_sample['answers'])
            answer_obj = qa_sample['answers'][ind]
            for rule_name, func in ANSWER_RULES:
                answer = func(answer_obj, a_toks, question, determiner=determiner)
                if answer:
                    answer = str(answer)
                    break
            else:
                raise ValueError('Missing answer')
        else:
            answer = get_unknown_answer(question)

        ques_parse = self.corenlp_cache[question]
        ques_tokens = ques_parse['tokens']
        ques_content_phrases = extract_phrases(ques_tokens)
        ques_const_parse = read_const_parse(ques_parse['parse'])
        q_list = alter_question(
            question, ques_tokens, ques_const_parse, self.nearby_word_dict,
            self.postag_dict, strategy='high-conf')

        sent_counter = 0
        altered_qa_samples = []
        for q_str, q_tokens, q_const_parse, tag in q_list:
            for rule in CONVERSION_RULES:
                sent = rule.convert(q_str, answer, q_tokens, q_const_parse)
                if sent:
                    sent_counter += 1
                    print(('  Choice %d: Distractor Sentence (%s): %s' % (sent_counter, tag, colored(sent, 'cyan'))))
                    if semantic_diff_check(corenlp.rejoin(ques_tokens), sent, ques_content_phrases):
                        pass
                    else:
                        print(colored("Not semantically different enough:", 'red'))
                        continue

                    altered_qa_sample = {
                        'question': qa_sample['question'],
                        'id': '%s-%s' % (qa_sample['id'], tag),
                        'answers': qa_sample['answers']
                    }

                    if self.version2:
                        altered_qa_sample['is_impossible'] = False

                    sentence_spans = [[s['tokens'][0]['characterOffsetBegin'], s['tokens'][-1]['characterOffsetEnd']]
                                      for s in context_parse['sentences']]
                    if paraphrase:
                        cur_text, answers, sentence_spans = self.replace_with_paraphrase(qa_sample, context, paraphrase['distractor'], sentence_spans, ans_loc)
                    else:
                        cur_text = context
                        answers = qa_sample['answers']

                    cur_text, new_answers, insert_span = insert_sentence(cur_text, answers, sent, sentence_spans)
                    altered_qa_sample['answers'] = new_answers
                    altered_qa_sample['adversarial_span'] = insert_span

                    if not cur_text or not new_answers:
                        continue

                    altered_paragraph = {'context': cur_text, 'qas': [altered_qa_sample]}
                    altered_qa_samples.append(altered_paragraph)
                    altered = True
                    break
        if altered:
            return altered_qa_samples
        else:
            return None

    def invalidate_answer(self, qa_sample, context, **kwargs):

        if qa_sample['answers'] == []:
            return None

        altered = False
        question = qa_sample['question'].strip()
        print(('Question: %s' % question))
        context_parse = self.corenlp_cache[context]
        ind, a_toks, ans_loc = get_tokens_for_answers(qa_sample['answers'], context_parse)

        ques_parse = self.corenlp_cache[question]
        ques_tokens = ques_parse['tokens']
        ques_content_phrases = extract_phrases(ques_tokens)

        ans_parse = self.corenlp_cache[context]['sentences'][ans_loc]
        a_toks_idxs = [] # to make sure that the answer is altered during alterations
        a_list = alter_answer(ans_parse, a_toks_idxs, self.nearby_word_dict, self.postag_dict, strategy='high-conf')

        sent_counter = 0
        altered_qa_samples = []
        for q_str, q_tokens, tag in a_list:
            sent = q_str
            if sent:
                sent_counter += 1
                print(('  Choice %d: Distractor Sentence (%s): %s' % (sent_counter, tag, colored(sent, 'cyan'))))

                if semantic_diff_check(corenlp.rejoin(ques_tokens), sent, ques_content_phrases):
                    pass
                else:
                    print(colored("Not semantically different enough:", 'red'))
                    continue

                orig_start_offset = ans_parse['tokens'][0]['characterOffsetBegin']
                orig_end_offset = ans_parse['tokens'][-1]['characterOffsetEnd']
                cur_text = context[:orig_start_offset] + sent + context[orig_end_offset:]
                new_answers = []

                altered_qa_sample = {
                    'question': qa_sample['question'],
                    'id': '%s-%s' % (qa_sample['id'], tag),
                    'answers': new_answers
                }
                if self.version2:
                    altered_qa_sample['is_impossible'] = True

                altered_paragraph = {'context': cur_text, 'qas': [altered_qa_sample]}
                altered_qa_samples.append(altered_paragraph)
                altered = True
                break

        if altered:
            return altered_qa_samples
        else:
            return None

    def replace_with_paraphrase(self, qa_sample, context, paraphrase, sentence_spans, answer_loc):

        start, end = sentence_spans[answer_loc]
        new_sentence_len = len(paraphrase)
        sent_len_diff = new_sentence_len - (end - start)
        new_sentence_spans = []
        for i in range(len(sentence_spans)):
            if i < answer_loc:
                new_sentence_spans.append(sentence_spans[i])
            elif i == answer_loc:
                new_end = end + sent_len_diff
                new_sentence_spans.append([start, new_end])
            else:
                new_sentence_spans.append([sentence_spans[i][0] + sent_len_diff, sentence_spans[i][1] + sent_len_diff])

        cur_text = context[:start] + paraphrase + context[end:]
        new_answers = []
        for a in qa_sample['answers']:

            if a['answer_start'] >= start and a['answer_start'] <= end:
                new_answer_start = paraphrase.find(a['text'])
                try:
                    assert new_answer_start >= 0, (a['text'], a['answer_start'], start, end)
                except:
                    continue
                new_answer_start += start
                assert cur_text[new_answer_start:new_answer_start + len(a['text'])] == a['text']
                new_answers.append({
                    'text': a['text'],
                    'answer_start': new_answer_start
                })

            elif a['answer_start'] < start:
                assert cur_text[a['answer_start']:a['answer_start'] + len(a['text'])] == a['text']
                new_answers.append({
                    'text': a['text'],
                    'answer_start': a['answer_start']
                })

            elif a['answer_start'] >= end:
                new_answer_start = a['answer_start'] + sent_len_diff
                assert cur_text[new_answer_start:new_answer_start + len(a['text'])] == a['text']
                new_answers.append({
                    'text': a['text'],
                    'answer_start': new_answer_start
                })
            else:
                print('Ran out of cases', a['answer_start'], start, end)

        return cur_text, new_answers, new_sentence_spans

    def paraphrase_wrapper(self, qa_sample, context, paraphrase):

        altered_qa_samples = []
        context_parse = self.corenlp_cache[context]
        altered_qa_sample = {
            'question': qa_sample['question'],
            'id': '%s-%s' % (qa_sample['id'], 'PerturbAnswer'),
            'answers': qa_sample['answers']
        }

        if self.version2:
            altered_qa_sample['is_impossible'] = False
        sentence_spans = [[s['tokens'][0]['characterOffsetBegin'], s['tokens'][-1]['characterOffsetEnd']]
                          for s in context_parse['sentences']]
        ind, a_toks, ans_loc = get_tokens_for_answers(qa_sample['answers'], context_parse)
        try:
            cur_text, new_answers, sentence_spans = self.replace_with_paraphrase(qa_sample, context, paraphrase['sentence'],
                                                                         sentence_spans, ans_loc)
        except KeyError:
            cur_text, new_answers, sentence_spans = self.replace_with_paraphrase(qa_sample, context,
                                                                                 paraphrase['distractor'],
                                                                                 sentence_spans, ans_loc)
        altered_qa_sample['answers'] = new_answers
        if not cur_text or not new_answers:
            return None

        altered_paragraph = {'context': cur_text, 'qas': [altered_qa_sample]}
        altered_qa_samples.append(altered_paragraph)
        return altered_qa_samples


class PerturbAnswer:

    def __init__(self, perturbation_file, syntactic_paraphrase_file=None):
        with open(perturbation_file, 'r') as f:
            lines = f.readlines()
        perturbations = {}
        for line in lines:
            perturbation = json.loads(line)
            perturbations[perturbation['id']] = perturbation
        self.perturbations = perturbations

        if syntactic_paraphrase_file is not None:
            with open(syntactic_paraphrase_file, 'r') as f:
                syn_perturbations = json.load(f)
        self.syntactic_paraphrases = syn_perturbations

    def generate_augmentation(self, id):
        if id in self.perturbations:
            perturbed_sentence = self.perturbations[id]
            return perturbed_sentence
        else:
            return None

    def generate_syntactic_paraphrase(self, id):
        paraphrase = None
        if id in self.syntactic_paraphrases:
            paraphrases = self.syntactic_paraphrases[id]
            for p, sim in paraphrases:
                if sim>= 0.99:
                    continue
                if sim <= 0.9:
                    break
                paraphrase = p
        return paraphrase


def test_augmentations(dataset):
    corenlp_cache = load_cache()
    nearby_word_dict = load_nearby_words()
    postag_dict = load_postag_dict()
    adversarial_attacker = AdversarialAttack(corenlp_cache=corenlp_cache,
                                             nearby_word_dict=nearby_word_dict,
                                             postag_dict=postag_dict)
    perturb_answer = PerturbAnswer(PERTURB_ANS_FILE.replace('squad_', 'dev_squad_'))

    for article in tqdm(dataset['data']):
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa_sample in paragraph['qas']:
                print(context)
                print(qa_sample)
                for neg_policy in ['AddSentDiverse', 'AddKSentDiverse', 'AddAnswerPosition', 'InvalidateAnswer']:
                    if qa_sample['answers'] != []:
                        paraphrase = perturb_answer.generate_augmentation(qa_sample['id'])
                    new_qa_samples = adversarial_attacker.generate_adversarial_sample(qa_sample, context, paraphrase=paraphrase, policy=neg_policy)
                    print('******', neg_policy, '*******')
                    if paraphrase:
                        print('Paraphrase: %s' % paraphrase['sentence'])
                    if new_qa_samples:
                        print(random.choice(new_qa_samples))
                break
            break
#        break

corenlp_cache = load_cache()
def dump_data(dataset, prefix, policy, outdir = None, keep_original=False, use_answer_placeholder=False, alteration_strategy=None):

    #corenlp_cache = load_cache()
    nearby_word_dict = load_nearby_words()
    postag_dict = load_postag_dict()
    adversarial_attacker = AdversarialAttack(corenlp_cache=corenlp_cache,
                                             nearby_word_dict=nearby_word_dict,
                                             postag_dict=postag_dict)
    perturb_answer = PerturbAnswer(PERTURB_ANS_FILE.replace('squad_', 'train_squad_'),
                                   syntactic_paraphrase_file=PERTURB_QUES_FILE.replace('squad_', 'train_squad_'))

    adversarial_count = 0
    out_data = []
    out_obj = {'version': dataset['version'], 'data': out_data}
    counter = 0
    for article in tqdm(dataset['data']):
        counter += 1
        out_paragraphs = []
        out_article = {'title': article['title'], 'paragraphs': out_paragraphs}
        out_data.append(out_article)
        for paragraph in article['paragraphs']:
            if keep_original:
                out_paragraphs.append(paragraph)
            context = paragraph['context']
            for qa_sample in paragraph['qas']:

                sub_policy = random.choice(policy)
                combined_policy, prob = sub_policy
                if '-' in combined_policy:
                    neg_policy, pos_policy = combined_policy.split('-')
                else:
                    if combined_policy == 'PerturbAnswer' or combined_policy == 'PerturbQuestion':
                        pos_policy = combined_policy
                        neg_policy = 'None'
                    else:
                        pos_policy = 'None'
                        neg_policy = combined_policy
                print(neg_policy, pos_policy, prob)

                if random.random() > prob:
                    continue
                if neg_policy is 'None' and pos_policy is 'None':
                    continue

                paraphrase = None
                q_paraphrase = None
                if pos_policy != 'None':
                    if qa_sample['answers'] != [] and pos_policy == 'PerturbAnswer':
                        paraphrase = perturb_answer.generate_augmentation(qa_sample['id'])
                    if pos_policy == 'PerturbQuestion':
                        q_paraphrase = perturb_answer.generate_syntactic_paraphrase(qa_sample['id'])
                if neg_policy != 'None':
                    new_qa_samples = adversarial_attacker.generate_adversarial_sample(qa_sample, context, paraphrase=paraphrase, policy=neg_policy)
                    if new_qa_samples:
                        adversarial_count += 1
                        candidate = random.choice(new_qa_samples)
                        if q_paraphrase:
                            candidate['qas'][0]['question'] = q_paraphrase
                        out_paragraphs.append(candidate)
                    else:
                        if paraphrase:
                            new_qa_samples = adversarial_attacker.paraphrase_wrapper(qa_sample, context, paraphrase)
                            adversarial_count += 1
                            out_paragraphs.append(random.choice(new_qa_samples))
                else:
                    if paraphrase:
                        new_qa_samples = adversarial_attacker.paraphrase_wrapper(qa_sample, context, paraphrase)
                        adversarial_count += 1
                        out_paragraphs.append(random.choice(new_qa_samples))
                    if q_paraphrase:
                            print("Paraphrase: %s" % q_paraphrase)
                            qa_sample['question'] = q_paraphrase
                            out_paragraphs.append({'context': paragraph['context'], 'qas': [qa_sample]})
                            
    prefix = 'train-%s' % prefix
    if not outdir:
        outdir = '../out'
    outfile = os.path.join(outdir, prefix + '.json')
    with open(outfile, 'w') as f:
        json.dump(out_obj, f)
    with open(outfile.replace('.json', '-indented.json'), 'w') as f:
        json.dump(out_obj, f, indent=2)

    return outfile

def augment_with_adversaries(policy, data_path, prefix='AutoAugment', keep_original=False):
    dataset = read_data(data_path)
    num_sub_policies = int(len(policy)/2)
    policy = [policy[i * 2:i * 2 + 2] for i in range(0, num_sub_policies)]
    augmented_file_path = dump_data(dataset, prefix, policy, outdir=os.path.dirname(data_path), alteration_strategy='high-conf', keep_original=keep_original)
    return augmented_file_path

if __name__ == '__main__':

    policy = ['AddSentDiverse', 'PerturbAnswer', 0.9,
              'AddSentDiverse', 'PerturbAnswer', 0.9,
              'AddSentDiverse', 'PerturbAnswer', 0.9,
              'AddSentDiverse', 'PerturbAnswer', 0.9,
              'AddSentDiverse', 'PerturbAnswer', 0.9]
    
    # AutoAugment (Controller Step 37, Reward = dev score, reduced dataset, valid_score = 80.15]
    policy = ["AddAnswerPosition", "None", 0.8,
              "InvalidateAnswer", "PerturbAnswer", 0.2,
              "None", "PerturbAnswer", 0.2,
              "InvalidateAnswer", "None", 0.4,
              "InvalidateAnswer", "None", 0]

    # AutoAugment (Controller Step 63, Reward = Dev score, reduced dataset, valid_score = 80.01]
    policy = ["AddSentDiverse", "PerturbAnswer", 0.4,
             "AddKSentDiverse", "PerturbAnswer", 0.8,
             "None", "None", 0.8,
             "InvalidateAnswer", "None", 0.2,
             "AddKSentDiverse", "PerturbAnswer", 0.4]
    
    policy = ['AddAnswerPosition', 0.04190139877173671, 
              'AddAnswerPosition-PerturbAnswer', 0.17418683157537918, 
              'AddAnswerPosition-PerturbQuestion', 0.5654020055647535, 
              'AddKSentDiverse',  0.17289117059986037, 
              'AddKSentDiverse-PerturbAnswer', 0.5666104779444462, 
              'AddSentDiverse', 0.5140042529495411, 
              'AddSentDiverse-PerturbAnswer', 0.8688708875473768, 
              'AddSentDiverse-PerturbQuestion', 0.7203472479309528, 
              'PerturbAnswer', 0.9035913470702983, 
              'PerturbQuestion', 0.2776149982333721]

    #data_path = augment_with_adversaries(policy, data_path='../AutoAdverse/data/train-sampled-v1.1.json', prefix='BayesAugment-S2G', keep_original=True)
    #print(data_path)
    
    policy = ['AddAnswerPosition', 1.0, 
              'AddSentDiverse', 1.0,
              'InvalidateAnswer', 1.0, 
              'PerturbAnswer', 1.0, 
              'AddKSentDiverse', 1.0, 
              'PerturbQuestion', 1.0]
    
    dataset = augment_with_adversaries(policy, data_path='../AutoAdverse/data/train-v2.0.json', prefix='MixedAdversary', keep_original=True)