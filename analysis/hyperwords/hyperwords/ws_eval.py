#!/usr/bin/env python2

from docopt import docopt
from scipy.stats.stats import spearmanr
import scipy, numpy as np

from representations.representation_factory import create_representation


def main():
    args = docopt("""
    Usage:
        ws_eval.py [options] <representation> <representation_path> <task_path>

    Options:
        --neg NUM     Number of negative samples; subtracts its log from PMI (only applicable to PPMI) [default: 1]
        --w+c         Use ensemble of word and context vectors (not applicable to PPMI)
        --eig NUM     Weighted exponent of the eigenvalue matrix (only applicable to SVD) [default: 0.5]
        --vocab FILE  Optional: use vocabulary file to determine what is difficult for the embeddings
        --cutoff NUM  Optional: Cutoff proportion for reporting rank mismatches
        --verbose NUM Specify 1 for bonus output for analysis
    """)

    data = read_test_set(args['<task_path>'])
    representation = create_representation(args)
    #print dir(representation), representation.iw[:3]
    correlation, actual, expected = evaluate(representation, data)
    top_n = 50
    print args['<representation>'], args['<representation_path>'], '\t%0.6f' % correlation
    #print args['--verbose']
    verbose = 1 if args['--verbose'] is not None and args['--verbose'] == '1' else 0
    if args['--vocab'] is not None:
        reconstruct_spearmanr(actual, expected, representation, data, args['--vocab'], cutoff=args['--cutoff'], verbose=verbose)

def read_test_set(path):
    test = []
    with open(path) as f:
        for line in f:
            x, y, sim = line.strip().lower().split()
            test.append(((x, y), sim))
    return test


def evaluate(representation, data):
    results = []
    for (x, y), sim in data:
        results.append((representation.similarity(x, y), float(sim)))
    actual, expected = zip(*results)
    return spearmanr(actual, expected)[0], actual, expected


def reconstruct_spearmanr(actual, expected, representation, data, vocab_file, cutoff=None, verbose=0):
    #print "verbose is ", verbose, " and equals 1?", verbose == 1
    #print type(verbose)
    # Make vocab dictionary
    if vocab_file is not None:
        with open(vocab_file, 'r') as f:
            vlines = {}
            for line in f.readlines():
                x,y = line.rstrip().split(' ')
                vlines[x] = int(y)
    # Reproduce rank generation
    results = []
    pairing = []
    for (x,y), sim in data:
        results.append((representation.similarity(x,y), float(sim)))
        pairing.append((x,y))
    results = np.asarray(results)
    pairing = np.asarray(pairing)
    m_vars = results.shape[1]
    m_obs = results.shape[0]
    # Reproduce scoring
    a, axisout = scipy.stats.stats._chk_asarray(actual, 0)
    b, _ = scipy.stats.stats._chk_asarray(expected, 0)
    a = np.column_stack((a,b))
    n_vars = a.shape[1]
    n_obs = a.shape[0]
    a_contains_nan, nan_policy = scipy.stats.stats._contains_nan(a, 'propagate')
    variable_has_nan = np.zeros(n_vars, dtype=bool)
    if a_contains_nan:
        if a.ndim == 1 or n_vars <= 2:
            return (np.nan, np.nan)
        else:
            variable_has_nan = np.isnan(a).sum(axis=0)
    a_ranked = np.apply_along_axis(scipy.stats.stats.rankdata, 0, a)
    #print a_ranked.shape
    # Calculate error (new)
    rank_diff = np.abs(a_ranked[:,0] - a_ranked[:,1])
    rel_rank_diff = a_ranked[:,0] - a_ranked[:,1]
    n_ranks = len(rank_diff)
    argsort = np.argsort(rank_diff)
    vrank = list(vlines.keys())
    vtotal = sum(list(vlines.values()))
    overrank = 0
    underrank = 0
    for arg in argsort[::-1]:
        if vocab_file is not None:
            try:
                word1, word2 = pairing[arg]
                count1, count2 = vlines[word1], vlines[word2]
                wordrank1, wordrank2 = vrank.index(word1), vrank.index(word2)
                wordratio1, wordratio2 = count1/vtotal, count2/vtotal
                if verbose == 1 and cutoff is not None and rank_diff[arg] < float(cutoff):
                    print "rank diff", rank_diff[arg], "Cutoff at ", cutoff, "proportionate rank difference"
                    break
                '''
                print arg, word1, count1, "(rank ", wordrank1, ", ratio", wordratio1, ")", \
                           word2, count2, "(rank ", wordrank2, ", ratio", wordratio2, ")", \
                           " has rank difference ", rank_diff[arg], " based on results ", results[arg]
                '''
                # For aggressiveness see which way embedding needs to go
                if rel_rank_diff[arg] < 0:
                    diff = "not similar enough"
                    underrank = underrank + 1
                    rank_indicator = 0
                else:
                    diff = "too similar"
                    overrank = overrank + 1
                    rank_indicator = 1
                # For binning purposes see word counts
                if verbose == 1:
                    print count1, count2, rank_indicator
                    #print "Word counts:", count1, count2, "and embedding is", diff
                    #print arg, " ", pairing[arg], " ", vlines[pairing[arg][0]], " ", vlines[pairing[arg][1]], " ", rank_diff[arg], " ", results[arg]
            except KeyError:
                try:
                    word = vlines[pairing[arg][0]]
                except KeyError:
                    word = pairing[arg][0]
                else:
                    word = pairing[arg][1]
                finally:
                    pass
                    #if verbose == 1
                        #print word, " is not in vocabulary! skipping pair ", pairing[arg]
        else:
            if verbose == 1:
                print arg, " ", pairing[arg], " ", rank_diff[arg], " ", results[arg]
    #print "Overrank count: ", overrank, "Underrank count: ", underrank
    #print a_ranked[:5], rank_diff[:5]
    #print np.max(rank_diff), np.mean(rank_diff), np.min(rank_diff)
    # Get spearmanr (old)
    rs = np.corrcoef(a_ranked, rowvar=0)
    dof = n_obs-2
    with np.errstate(divide='ignore'):
        t = rs * np.sqrt((dof/((rs+1.0)*(1.0-rs))).clip(0))
    prob = 2 * scipy.stats.distributions.t.sf(np.abs(t),dof)
    if rs.shape == (2,2):
        #print 'quick out'
        return (rs[1,0], prob[1,0])
    else:
        #print 'slow out'
        rs[variable_has_nan,:] = np.nan
        rs[:, variable_has_nan] = np.nan
        return (rs, prob)

if __name__ == '__main__':
    main()
