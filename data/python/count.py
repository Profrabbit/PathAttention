import json
from tqdm import tqdm
code_count = dict()
with open('./test.json') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        data = json.loads(line)
        l = len(data['content'])
        if l in code_count:
            code_count[l] += 1
        else:
            code_count[l] = 1
    _keys = sorted(code_count.keys())
    _c = 0
    _s = 0
    for k in _keys:
        _s += code_count[k]

    for k in _keys:
        _c += code_count[k]
        print('len of code <={} has {}'.format(k, _c / _s))
