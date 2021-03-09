from collections import Counter
import os

true_positive, false_positive, false_negative = 0, 0, 0
dir = 'history_run/relation_python_2021-03-05-01-14-36'
with open(os.path.join(dir, 'pred.txt'), 'r') as f1, open(os.path.join(dir, 'ref.txt'), 'r') as f2:
    predict = f1.readlines()
    original = f2.readlines()
    pre, rec, f = 0, 0, 0
    for p, o in zip(predict, original):
        p, o = p.split(), o.split()
        common = Counter(p) & Counter(o)
        true_positive = sum(common.values())
        false_positive = (len(p) - sum(common.values()))
        false_negative = (len(o) - sum(common.values()))

        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        pre += precision
        rec += recall
        f += f1
    print(pre / len(predict), rec / len(predict), f / len(predict))
# 0.2562899130568593 0.1953627495063635 0.22171684675290096
# 0.26144458389232533 0.20127120767078346 0.21546496184484706
