import os
import glob
import numpy as np
import cPickle

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction import DictVectorizer
from pulp import *

from read_data import make_data_dict, make_mention_dict, get_corefs
from features import featurize

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

    Returns tuple of (train_samples, data_dict) for each part of
    each document in `data_dir`. `train_samples` are pairs of
    4-tuples in the format of tuples created by `read_data.get_corefs.`

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

def fvec(i, j, d, v):
    """Given 4-tuple representations of mentions, return onehot feat vec
    for that pair."""
    m_i = make_mention_dict(i, d)
    m_j = make_mention_dict(j, d)
    feat_dict = featurize(m_i, m_j, d)
    fv = v.transform(feat_dict)
    return fv

def generate_links(model, data_dict, vectorizer= None,
                   first_match=True, ilp=False):
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

    #TODO clean up this horribly ugly code.
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
    """ Given a list of mention pairs determined to be coreferent,
    create a partitioning of the set of entities (i.e., number the
    coreference chains.
    """
    #TODO Fix the logic here. Although the method (seemingly) works,
    # there is an error in the implementation which forces the terrible
    # brute force approach here.
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

def generate_links_file(model, test_doc, vectorizer=None, first_match=True, ilp=False):
    """ Returns list of lists of predicted coref links. Each list corresponds
    to a part of the document.
    """
    dicts = make_data_dict(test_doc)
    doc_links = []
    for d in dicts:
        doc_links.append(
            generate_links(model, d, vectorizer=vectorizer,
                 first_match=first_match, ilp=ilp))
    return doc_links

def write_all_test_output(data_dir, model, vectorizer,
                          first_match=True, ilp=False):
    test_docs = glob.glob(data_dir + '/*conll')
    for doc in tqdm(test_docs):
        doc_links = generate_links_file(model, doc, vectorizer=vectorizer,
                                        first_match=first_match, ilp=ilp)
        write_test_output(doc_links, doc)

def create_master_test_files(response_dir):
    """ Given the directory full of key and response files, generate
    a single master.response and master.key file for scoring.
    """
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
