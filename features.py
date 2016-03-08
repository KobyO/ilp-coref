import nltk.tree
from nltk.corpus import names
from nltk.corpus import wordnet as wn

""" Feature functions for pairwise coreference classifier. """

MALE_NAMES = names.words('male.txt')
FEMALE_NAMES = names.words('female.txt')

M_NAME_SET = set(MALE_NAMES)
F_NAME_SET = set(FEMALE_NAMES)

SING_PRONOUNS = ['i', 'you', 'he', 'she', 'it', 'me', 'him', 'her', 'his',
                 'my', 'myself', 'yourself', 'himself', 'herself', 'itself']
PL_PRONOUNS = ['you', 'we', 'they', 'us', 'them' 'yourselves',
               'ourselves', 'themselves']

def featurize(m_i, m_j, data_dict):
    """ `m_i` is candidate antecedent, `m_j` is anaphor,
    represented as mention_dicts.

    Categorical variables are turned into strings so they
    will be one-hot encoded by dictvectorizer later
    """
    i_str = m_i['string'].lower().strip('.,?!:;')
    j_str = m_j['string'].lower().strip('.,?!:;')
    fdict = {
        #'strmatch': strmatch(i_str, j_str),
        'word_str': word_str(m_i, m_j),
        'pro_str': pro_str(i_str, j_str),
        'prop_str': prop_str(m_i, m_j),
        'type_match': type_match(m_i, m_j),
        'dist': dist(m_i, m_j),
        'i_pron': 1 if (i_str in SING_PRONOUNS or \
                        i_str in PL_PRONOUNS) else 0,
        'j_pron': 1 if (j_str in SING_PRONOUNS or \
                        j_str in PL_PRONOUNS) else 0,
        #'i_n_words': n_words(i_str),
        #'j_n_words': n_words(j_str),
        'both_proper': check_both_proper(m_i, m_j),
        'subject_i': is_subject(m_i, data_dict),
        'j_defnp': def_np(j_str),
        'j_demnp': dem_np(j_str),
        'embedded': embedded(m_i, m_j),
        'gender_agree': gender_agr(m_i, m_j),
        'num_agree': num_agr(m_i, m_j),
        'agree': agree(m_i, m_j),
        #'maximal_np': maximal_np(m_i, m_j),
        'head_match': head_match(m_i, m_j),
        'animacy_match': animacy_match(m_i, m_j)
    }
    return fdict


def head(m):
    m_toks = m['string'].split()
    m_pos = m['pos'].split()
    if len(m_pos) == 1:
        m_head = m['string'].strip('.,?!:;')
        return m_head
    head_pos = ['POS', 'NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']
    for pos in head_pos:
        for mp in m_pos:
            if mp == pos:
                m_head = m_toks[m_pos.index(mp)].strip('.,?!:;')
                return m_head
    return '<unk>'

def animacy_match(m_i, m_j):
    i_head = head(m_i)
    j_head = head(m_j)
    a_i = animate(i_head)
    a_j = animate(j_head)
    if a_i == a_j:
        if a_i != 'unk':
            return '1'
        else:
            return 'unk'
    return '0'

animacy_cache = {}
def animate(m):
    if m.lower() in animacy_cache:
        return animacy_cache[m.lower()]
    if m in M_NAME_SET or m in F_NAME_SET:
        animacy_cache[m] = True
        return True
    anim_pros = [pro for pro in SING_PRONOUNS + PL_PRONOUNS]
    anim_pros.remove('it')
    anim_pros.remove('itself')
    if m.lower() in anim_pros:
        animacy_cache[m] = True
        return True
    hyper = lambda s: s.hypernyms()
    try:
        m_syn = wn.synsets(m, 'n')[0]
    except (UnicodeDecodeError, IndexError):
        animacy_cache[m] = 'unk'
        return 'unk'
    closure = m_syn.closure(hyper)
    if wn.synset('living_thing.n.01') in closure:
        animacy_cache[m] = True
        return True
    else:
        animacy_cache[m] = False
        return False
    animacy_cache[m] = 'unk'
    return 'unk'

def head_match(m_i, m_j):
    i_head = head(m_i)
    j_head = head(m_j)
    if not (j_head or i_head):
        return '0'
    elif j_head.lower() == i_head.lower():
        if j_head != '<unk>':
            return '1'
        return 'unk'
    else:
        return '0'

def maximal_np(m_i, m_j):
    if m_i['sent_num'] != m_j['sent_num']:
        return 0
    t = m_j['tree']
    i_str = m_i['string'].strip('.,?!:;')
    j_str = m_j['string'].strip('.,?!:;')
    maxnp = list(t.subtrees(filter=lambda x: x.label() == 'NP' and \
                i_str in x.leaves() and j_str in x.leaves()))
    if maxnp:
        return 1
    return 0

def agree(m_i, m_j):
    num = num_agr(m_i, m_j)
    gen = gender_agr(m_i, m_j)
    if num == '1' and gen == '1':
        return '1'
    elif num == '1' and gen == '0':
        return '0'
    else:
        return 'unk'

def strmatch(i_str, j_str):
    if truncate_str(i_str) == truncate_str(j_str):
        return 1
    return 0

def pronoun(m_str):
    if m_str in SING_PRONOUNS or m_str in PL_PRONOUNS:
        return True
    return False

def pro_str(i_str, j_str):
    if pronoun(i_str) and pronoun(j_str):
        if i_str == j_str:
            return 1
    return 0

def prop_str(m_i, m_j):
    i_str = m_i['string'].strip('.,?!:;')
    j_str = m_j['string'].strip('.,?!:;')
    if proper(i_str) and proper(j_str):
        if i_str == j_str:
            return 1
    return 0

def word_str(m_i, m_j):
    i_str = m_i['string'].strip('.,?!:;')
    j_str = m_j['string'].strip('.,?!:;')
    if not (pronoun(i_str.lower()) or pronoun(j_str.lower())):
        if not (proper(i_str) or proper(j_str)):
            if truncate_str(i_str).lower() == truncate_str(j_str).lower():
                return 1
    return 0

def truncate_str(string):
    dets = ['a', 'an', 'the', 'this', 'these', 'those', 'that']
    if string.split()[0].lower() in dets:
        return ' '.join(string.split()[1:])
    return string

def dist(m_i, m_j):
    d = abs(m_j['sent_num'] - m_i['sent_num'])
    if d <= 25:
        return str(d)
    elif 25 < d <= 50:
        return '26-50'
    elif 50 < d <= 75:
        return '51-75'
    else:
        return '>75'

def n_words(m_str):
    n = len(m_str.split())
    if n <= 25:
        return str(n)
    else:
        return '>25'

def type_match(m_i, m_j):
    if m_i['e_type'] and m_j['e_type']:
        if m_i['e_type'] == m_j['e_type']:
            return '1'
        return '0'
    elif m_i['e_type'] and not m_j['e_type']:
        return '0'
    elif not m_i['e_type'] and m_j['e_type']:
        return '0'
    return 'unk'

def def_np(j_str):
    return 1 if j_str.split()[0] == 'the' else 0

def dem_np(j_str):
    dems = ['this', 'that', 'these', 'those']
    return 1 if j_str.split()[0] in dems else 0

def proper(m_str):
    m_toks = m_str.strip('.,?!:;').split()
    for t in m_toks:
        if not t[0].isupper() or not t[1:].islower():
            return False
    return True

def check_both_proper(m_i, m_j):
    i_str = m_i['string'].strip('.,?!:;')
    j_str = m_j['string'].strip('.,?!:;')
    if proper(i_str) and proper(j_str):
        return 1
    return 0

def is_subject(m_i, d):
    subj = find_subject(m_i['tree'])
    if subj:
        if subj.leaves() == m_i['string'].strip('.,?!:;').split():
            return 1
    return 0

def find_subject(tree):
    if isinstance(tree,nltk.Tree):
        if tree.label() == 'TOP':
            return find_subject(tree.copy().pop())
        elif tree.label() == 'S':
            try:
                if 'NP' in tree[0].label() and 'VP' in tree[1].label():
                    return tree[0]
            except IndexError:
                return 0

def gender_agr(m_i, m_j):
    g_i = gender(m_i)
    g_j = gender(m_j)
    if g_i != 'unk' and g_j != 'unk':
        if g_i == g_j:
            return '1'
        else:
            return '0'
    else:
        return 'unk'

def gender(m):
    feminine = ['she', 'her', 'miss', 'mrs', 'ms', 'lady',
                'woman', 'girl', 'ma\'am']
    masculine = ['he', 'him', 'his', 'mr', 'sir', 'man', 'boy']
    m_head = head(m)
    if m_head.lower() in feminine:
        return 'fem'
    elif m_head.lower() in masculine:
        return 'masc'
    elif m_head in M_NAME_SET:
        return 'masc'
    elif m_head in F_NAME_SET:
        return 'fem'
    else:
        return 'unk'

def appositive(m_j):
    tokens = m_j['string'].split()
    a = list(t.subtrees(filter=lambda x:
                    (x.label() == 'NP' and len(x) == 3 and x[0].label() == 'NP'
                     and x[1].label() == ',' and x[2].label() == 'NP'
                     and [t for t in tokens if t in x[2].leaves()] is not None)))
    return 1 if a else 0

def number(m):
    toks = m['string'].lower().split()
    for tok in toks[::-1]:
        if tok in SING_PRONOUNS:
            return 'singular'
        elif tok in PL_PRONOUNS:
            return 'plural'
    pos_list = [pos.strip('.,?!:;') for pos in m['pos'].split()]
    p = ['NN', 'NNS', 'NNP', 'NNPS']
    pmatches = []
    for pos in pos_list[::-1]:
        if pos in p:
            if pos.endswith('S'):
                pmatches.append('plural')
            pmatches.append('singular')
    pset = set(pmatches)
    if len(pset) == 1:
        return pset.pop()
    else:
        return 'unk'

def num_agr(m_i, m_j):
    n_i = number(m_i)
    n_j = number(m_j)
    if n_i != 'unk' and n_j != 'unk':
        if n_i == n_j:
            return '1'
        else:
            return '0'
    return 'unk'

def embedded(m_i, m_j):
    m_i_sent = m_i['sent_num']
    m_j_sent = m_j['sent_num']
    m_i_start = m_i['start']
    m_j_start = m_j['start']
    m_i_end = m_i['end']
    m_j_end = m_j['end']
    if m_i_sent == m_j_sent:
        x = [(m_i_start, 's'), (m_i_end, 'e'), (m_j_start, 's'), (m_j_end, 'e')]
        x = sorted(x, key=lambda x: (x[0], x[1]))
        num_intervals = 0
        for i in x:
            if i[1] == 's':
                num_intervals += 1
                if num_intervals > 1:
                    return 1
            else:
                num_intervals -= 1
    return 0
