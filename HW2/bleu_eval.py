import math
import operator
import sys
import json
from functools import reduce 
def countN(cand, ref, n):
    cc = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(cand)):
        # Calculate precision for each sen
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in ref:
            ref_sen = reference[si]
            ngram_d = {}
            words = ref_sen.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # cand
        cand_sen = cand[si]
        cand_dict = {}
        words = cand_sen.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        cc += countClips(cand_dict, ref_counts)
        count += limits
        r += bestLengthMatch(ref_lengths, len(words))
        c += len(words)
    if cc == 0:
        pr = 0
    else:
        pr = float(cc) / count
    bp = penaltyBrevity(c, r)
    return pr, bp


def countClips(cand_d, ref_ds):
    """Count the clip count for each ngram considering all ref"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def bestLengthMatch(ref_l, cand_l):
    """Find the closest length of reference to that of cand"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def penaltyBrevity(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(s,t,flag = False):

    score = 0.  
    count = 0
    cand = [s.strip()]
    if flag:
        ref = [[t[i].strip()] for i in range(len(t))]
    else:
        ref = [[t.strip()]] 
    precisions = []
    pr, bp = countN(cand, ref, 1)
    precisions.append(pr)
    score = geometric_mean(precisions) * bp
    return score



if __name__ == "__main__" :
    test = json.load(open('MLDS_hw2_1_data/testing_label.json','r'))
    output = sys.argv[1]
    result = {}
    with open(output,'r') as f:
        for li in f:
            li = li.rstrip()
            comma = li.index(',')
            test_id = li[:comma]
            caption = li[comma+1:]
            result[test_id] = caption
    bleu=[]
    for item in test:
        each_scores = []
        caps = [x.rstrip('.') for x in item['caption']]
        each_scores.append(BLEU(result[item['id']],caps,True))
        bleu.append(each_scores[0])
    average = sum(bleu) / len(bleu)
    print("Avg bleu score = " + str(average))
