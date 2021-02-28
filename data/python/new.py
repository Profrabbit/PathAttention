import pygments.lexers
import json
from tqdm import tqdm

type_dic = dict()
with open('./data/valid.json') as f1:
    lines_1 = f1.readlines()
    for line_1 in tqdm(lines_1):
        content = json.loads(line_1)['content']
        for token in content:
            temp_code = '_'.join(token)
            for _t in pygments.lex(temp_code, pygments.lexers.PythonLexer()):
                a = _t[0]
                type_l = str(a).split('.')
                type_ = type_l[1]
                if type_ in type_dic:
                    type_dic[type_] += 1
                else:
                    type_dic[type_] = 1
                break
print(type_dic)
