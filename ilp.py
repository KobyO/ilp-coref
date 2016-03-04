import os
import glob
import numpy as np
import nltk.tree
import cPickle
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction import DictVectorizer
from pulp import *

from read_data import make_data_dict, make_mention_dict
from read_data import get_corefs, get_all_mentions

data_dir = '../data/train'

SING_PRONOUNS = ['i', 'you', 'he', 'she', 'it', 'me', 'him', 'her', 'his',
                 'my', 'myself', 'yourself', 'himself', 'herself', 'ourselves']
PL_PRONOUNS = ['you', 'we', 'they', 'us', 'them' 'yourselves',
               'ourselves', 'themselves']

def make_train_instances(data_dir):
    """
    Training instances are constructed using pairs (i, j) where
    j is an anaphor and i is a candidate antecedent. Each pair
    is labeled either 1 for coreferent, else 0. Only immediately
    adjacent pairs in a coreference chain are used to generate
    positive examples. The first NP in the pair is the antecedent,
    and the second is the anaphor. Negative examples are formed by
    pairing each mention in between i and j that is either not
    coreferent with j.

    """
    train_files = glob.glob(data_dir+'/*.v4_auto_conll')
    dataset = []
    all_dicts = []
    for train_file in tqdm(train_files):
        dicts = make_data_dict(train_file)
        for d in dicts:
            instances = []
            c = get_corefs(d)
            # m = get_all_mentions(d, c)
            rev_c = c[::-1] # reverse c to start from end of document
            for idx, j in enumerate(rev_c):
                for i in rev_c[idx+1:]:
                    if i[-1] == j[-1]: # match
                        instances.append((i, j, True))
                        break
                    else:
                        instances.append((i, j, False))
            dataset.append(instances)
            all_dicts.append(d)
    return zip(dataset, all_dicts)

def make_dataset(training_samples):
    """Make dataset usable by sklearn."""
    # X: array (n_features, n_samples)
    # y: array (n_samples,), 1 if coreferent
    X = []
    y = []
    for (doc_part, d) in tqdm(training_samples):
        for pair in doc_part:
            i = pair[0]
            j = pair[1]
            label = 1 if pair[2] else 0
            m_i = make_mention_dict(i, d)
            m_j = make_mention_dict(j, d)
            feat_dict = featurize(m_i, m_j, d)
            X.append(feat_dict)
            y.append(label)
    print "Vectorizing feature dicts..."
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(X)
    y = np.array(y)
    return X, y, v

def build_model(X, y):
    # find optimal regularization strength
    cs = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    lr_cv = LogisticRegressionCV(Cs=cs, solver='liblinear', verbose=1)
    lr_cv.fit(X, y)
    # train pairwise maxent coref classifier
    lr = LogisticRegression(C=lr_cv.C_, random_state=0)
    lr = lr.fit(X, y)
    return lr

def test_file(model, test_doc, vectorizer=None, first_match=True, ilp=False):
    """ Returns list of lists of predicted coref links. Each list corresponds
    to a part of the document.
    """
    dicts = make_data_dict(test_doc)
    doc_links = []
    for d in dicts:
        doc_links.append(
            test(model, d, vectorizer=vectorizer,
                 first_match=first_match, ilp=ilp))
    return doc_links

def test(model, data_dict, vectorizer= None, first_match=True, ilp=False):
    # for each mention, work backwards and add a link for all previous
    # which the clasifier deems coreferent (first_match=False)
    #
    # for each mention, work backwards until you find a previous mention
    # which the classifier thinks is coreferent, add a link, and terminate
    # the search (first_match=True)
    d = data_dict
    c = get_corefs(d)
    v = vectorizer
    links = []
    rev_c = c[::-1]

    if ilp:
        # reverse the list of mentions to be in correct order for testing
        rev_c_idx = [(i,m) for i,m in enumerate(rev_c)]
        # cartesian product of mentions where m_i > m_j, i.e.,
        # m_i occurs in doc before m_j
        pairs = [(m_i, m_j) for m_j in rev_c_idx for m_i in rev_c_idx
                 if m_i > m_j]
        # dict of i_j: featurevector
        m_idx = {str(i[0])+'_'+str(j[0]): fvec(i[1], j[1], d, v) for i,j in pairs}
        # list of x_ij variable string names
        x_vars = [ij for ij in m_idx.keys()]
        # dict of i_j: p(i_j)
        p = {ij: model.predict_proba(fv)[0][1] for ij, fv in m_idx.items()}
        log_p = {ij: np.log(pr) for ij, pr in p.items()}
        log_not_p = {ij: np.log(1-pr) for ij, pr in p.items()}
        x = LpVariable.dicts("x", x_vars, cat='Binary')
        problem = LpProblem('coref', LpMaximize)
        problem += lpSum([log_p[idx] * x[idx] + log_not_p[idx] * x[idx]
                          for idx in m_idx.keys()])
        # add transitivty constraint
        trips = []
        for ij in x:
            for jk in x:
                if ij.split('_')[1] == jk.split('_')[0]:
                    trips.append((ij,jk,ij.split('_')[0]+'_'+jk.split('_')[1]))
        for ij,jk,ik in trips:
            problem += (1-x[ij]) + (1-x[jk]) >= (1-x[ik])

    else:
        rev_c_idx = [(i,m) for i,m in enumerate(rev_c)]
        for idx, j in rev_c_idx:
            for idx2, i in rev_c_idx[idx+1:]:
                X = fvec(i, j, d, v)
                ypred = model.predict(X)[0]
                if ypred == 1:
                    links.append((i,j))
                    if first_match:
                        break
                    else:
                        continue
    return links

def partition_links(links):
    subpart = []
    seen = []
    for link in links:
        if len(subpart) == 0:
            subpart.append([link[0],link[1]])
            seen.append(link[0])
            seen.append(link[1])
        else:
            in_subpart = False
            for i,part in enumerate(subpart):
                if link[0] in part and link[1] not in part:
                    if link[1] not in seen:
                        part.append(link[1])
                        seen.append(link[1])
                        in_subpart = True
                        break
                elif link[0] not in part and link[1] in part:
                    if link[0] not in seen:
                        part.append(link[0])
                        seen.append(link[0])
                        in_subpart = True
                        break
                elif link[0] in part and link[1] in part:
                    in_subpart = True
                    break
            if not in_subpart:
                if not link[0] in seen and not link[1] in seen:
                    subpart.append([link[0],link[1]])
                    seen.append(link[0])
                    seen.append(link[1])
    return subpart

def write_all_test_output(data_dir, model, vectorizer,
                          first_match=True, ilp=False):
    test_docs = glob.glob(data_dir + '/*conll')
    for doc in tqdm(test_docs):
        doc_links = test_file(model, doc, vectorizer=vectorizer,
                              first_match=first_match, ilp=ilp)
        write_test_output(doc_links, doc)

def create_test_files(response_dir):
    responses = glob.glob(response_dir + '/*.response')
    keys = glob.glob(response_dir + '/*.key')
    assert len(responses) == len(keys)
    docs = zip(sorted(keys), sorted(responses))
    with open(os.path.join(response_dir, 'master.response'), 'w+') as r:
        with open(os.path.join(response_dir, 'master.key'), 'w+') as k:
            for sk,sr in docs:
                assert os.path.basename(sk).split('.')[0] == \
                    os.path.basename(sr).split('.')[0]
                with open(sk) as kf, open(sr) as rf:
                    klines = kf.readlines()
                    rlines = rf.readlines()
                    for kl in klines:
                        k.write(kl)
                    for rl in rlines:
                        r.write(rl)

def write_test_output(doc_links, test_doc):
    partitions = []
    for linkset in doc_links:
        p = partition_links(linkset)
        partitions.append(p)
    out = []
    with open(test_doc) as doc, \
        open(os.path.join('responses',os.path.basename(test_doc)+'.key'),'w+') as key:
        part_sent = 0
        for line in doc:
            if line[0].startswith('#') and line.split()[-1].isdigit():
                out.append(line)
                key.write(line)
                part_sent = 0
            elif line[0].startswith('#'):
                out.append(line)
                key.write(line)
            elif line[0] == '\n':
                out.append(line.split())
                part_sent += 1
                key.write(line)
            else:
                line = line.split()
                part = line[1]
                word = line[2]
                #outline = line[:-1]
                outline = [line[0]]
                tmpline = [line[0], line[-1]]
                key.write('\t'.join(tmpline)+'\n')
                coref = []
                for i,p in enumerate(partitions[int(part)]):
                    for link in p:
                        if link[0] == part_sent:
                            start = link[1]
                            end = link[2]
                            word = int(word)
                            if start == word and end == word:
                                coref.append('({})'.format(i))
                            elif start == word:
                                coref.append('({}'.format(i))
                            elif end == word:
                                coref.append('{})'.format(i))
                if coref:
                    outline.append('|'.join(coref))
                else:
                    outline.append('-')
                out.append(outline)
    outfile = os.path.basename(test_doc)
    with open(os.path.join('responses',outfile+'.response'), 'w+') as outf:
        for line in out:
            if isinstance(line, list):
                outf.write('\t'.join(line)+'\n')
            else:
                outf.write(line)

def fvec(i, j, d, v):
    """Given 4-tuple representations of mentions, return onehot feat vec
    for that pair."""
    m_i = make_mention_dict(i, d)
    m_j = make_mention_dict(j, d)
    feat_dict = featurize(m_i, m_j, d)
    fv = v.transform(feat_dict)
    return fv

def featurize(m_i, m_j, data_dict):
    """ `m_i` is candidate antecedent, `m_j` is anaphor,
    represented as mention_dicts.

    Categorical variables are turned into strings so they
    will be one-hot encoded by dictvectorizer later
    """
    i_str = m_i['string'].lower()
    j_str = m_j['string'].lower()
    fdict = {
        #'strmatch': strmatch(i_str, j_str),
        'word_str': word_str(i_str, j_str),
        'pro_str': pro_str(i_str, j_str),
        'pn_str': pn_str(i_str, j_str),
        'type_match': type_match(m_i, m_j),
        'dist': dist(m_i, m_j),
        'i_pron': 1 if (i_str in SING_PRONOUNS or \
                        i_str in PL_PRONOUNS) else 0,
        'j_pron': 1 if (j_str in SING_PRONOUNS or \
                        j_str in PL_PRONOUNS) else 0,
        'i_n_words': n_words(i_str),
        'j_n_words': n_words(j_str),
        'both_proper': check_both_proper(m_i, m_j),
        'subject_i': is_subject(m_i, data_dict),
        'j_defnp': def_np(j_str),
        'j_demnp': dem_np(j_str),
        'embedded': embedded(m_i, m_j),
        'gender_agree': gender_agr(i_str, j_str),
        'num_agree': num_agr(m_i, m_j),
        'agree': agree(m_i, m_j),
        #'maximal_np': maximal_np(m_i, m_j)
    }
    return fdict

def maximal_np(m_i, m_j):
    if m_i['sent_num'] != m_j['sent_num']:
        return 0
    t = m_j['tree']
    i_str = m_i['string']
    j_str = m_j['string']
    maxnp = list(t.subtrees(filter=lambda x: x.label() == 'NP' and \
                i_str in x.leaves() and j_str in x.leaves()))
    if maxnp:
        return 1
    return 0

def agree(m_i, m_j):
    num = num_agr(m_i, m_j)
    gen = gender_agr(m_i['string'].lower(), m_j['string'].lower())
    if num == 1 and gen == '1':
        return '1'
    elif num == 0 and gen == '0':
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

def pn_str(i_str, j_str):
    if proper(i_str) and proper(j_str):
        if i_str == j_str:
            return 1
    return 0

def word_str(i_str, j_str):
    if not (pronoun(i_str) or pronoun(j_str)):
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
    m_toks = m_str.split()
    for t in m_toks:
        if not t[0].isupper() or not t[1:].islower():
            return False
    return True

def check_both_proper(m_i, m_j):
    i_str = m_i['string']
    j_str = m_j['string']
    if proper(i_str) and proper(j_str):
        return 1
    return 0

def is_subject(m_i, d):
    subj = find_subject(m_i['tree'])
    if subj:
        if subj.leaves() == m_i['string'].split():
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

def gender_agr(i_str, j_str):
    g_i = gender(i_str)
    g_j = gender(j_str)
    if g_i != 'unk' and g_j != 'unk':
        if g_i == g_j:
            return '1'
        else:
            return '0'
    else:
        return 'unk'

def gender(m_str):
    feminine = ['she', 'her', 'miss', 'mrs.', 'ms.', 'lady',
                'woman', 'girl', 'ma\'am']
    masculine = ['he', 'him', 'his', 'mr.', 'sir', 'man', 'boy']
    if any(s in feminine for s in m_str.split()):
        return 'fem'
    elif any(s in masculine for s in m_str.split()):
        return 'masc'
    else:
        return 'unk'

def appositive(m_j):
    tokens = m_j['string'].split()
    a = list(t.subtrees(filter=lambda x:
                    (x.label() == 'NP' and len(x) == 3 and x[0].label() == 'NP'
                     and x[1].label() == ',' and x[2].label() == 'NP')))
    return 1 if a else 0

def number(m):
    toks = m['string'].split()
    for tok in toks[::-1]:
        if tok in SING_PRONOUNS:
            return 'singular'
        elif tok in PL_PRONOUNS:
            return 'plural'
    pos_list = [pos.strip() for pos in m['pos'].split()]
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
