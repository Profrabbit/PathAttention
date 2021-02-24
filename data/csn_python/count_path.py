import json
from tqdm import tqdm

a, b = 0, 0
c = 0
with open('data/train.json') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        c += 1
        data = json.loads(line)
        a += len(data['paths'])
        b += len(data['paths_map'])
print(a / c, b / c)

# 1000 * 64
# 512*512 * 64
# 155.65  742.122
