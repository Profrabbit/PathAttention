import argparse
import os
import pickle as pkl
from tqdm import tqdm
import json
import ast
import itertools
import re
from copy import deepcopy


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


METHOD_NAME, NUM, Str = '<METHODNAME>', '<NUM>', '<STR>'
USED = 'USED'

inter_dic_path = './data/path_dic.pkl'
path_dic = dict()
if os.path.exists(inter_dic_path):
    with open(inter_dic_path, 'rb') as f:
        print('Already exist inter node dic')
        path_dic = pkl.load(f)

source_dic = dict()
target_dic = dict()


def parse_code(code_str):
    global c, d
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        raise Exception
    else:
        pass

    json_tree = []

    def gen_identifier(identifier, node_type='identifier'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        json_node['value'] = identifier
        return pos

    def traverse_list(l, node_type='list'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        children = []
        for item in l:
            children.append(traverse(item))
        if (len(children) != 0):
            json_node['children'] = children
        return pos

    def traverse(node):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = type(node).__name__
        children = []
        if isinstance(node, ast.Name):
            json_node['value'] = node.id
        elif isinstance(node, ast.Num):
            json_node['value'] = str(node.n)
        elif isinstance(node, ast.Str):
            json_node['value'] = node.s
        elif isinstance(node, ast.alias):
            json_node['value'] = str(node.name)
            if node.asname:
                children.append(gen_identifier(node.asname))
        elif isinstance(node, ast.FunctionDef):
            json_node['value'] = str(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            json_node['value'] = str(node.name)
        elif isinstance(node, ast.ClassDef):
            json_node['value'] = str(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                json_node['value'] = str(node.module)
        elif isinstance(node, ast.Global):
            for n in node.names:
                children.append(gen_identifier(n))
        elif isinstance(node, ast.arg):
            json_node['value'] = str(node.arg)
        elif isinstance(node, ast.NameConstant):
            json_node['value'] = str(node.value)

        # Process children.
        if isinstance(node, ast.For):
            children.append(traverse(node.target))
            children.append(traverse(node.iter))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.If) or isinstance(node, ast.While):
            children.append(traverse(node.test))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.With):
            # if node.context_expr:
            #     children.append(traverse(node.context_expr))
            # if node.optional_vars:
            #     children.append(traverse(node.optional_vars))
            children.append(traverse_list(node.items, 'items'))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.Try):
            children.append(traverse_list(node.body, 'body'))
            if node.handlers:
                children.append(traverse_list(node.handlers, 'handlers'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
            if node.finalbody:
                children.append(traverse_list(node.finalbody, 'finalbody'))
        # elif isinstance(node, ast.TryExcept):
        #     children.append(traverse_list(node.body, 'body'))
        #     children.append(traverse_list(node.handlers, 'handlers'))
        #     if node.orelse:
        #         children.append(traverse_list(node.orelse, 'orelse'))
        # elif isinstance(node, ast.TryFinally):
        #     children.append(traverse_list(node.body, 'body'))
        #     children.append(traverse_list(node.finalbody, 'finalbody'))

        elif isinstance(node, ast.arguments):
            children.append(traverse_list(node.args, 'args'))
            children.append(traverse_list(node.defaults, 'defaults'))
            if node.vararg:
                # children.append(gen_identifier(node.vararg, 'vararg'))
                children.append(traverse_list([node.vararg], 'vararg'))
            if node.kwarg:
                # children.append(gen_identifier(node.kwarg, 'kwarg'))
                children.append(traverse_list([node.kwarg], 'kwarg'))
        elif isinstance(node, ast.ExceptHandler):
            if node.type:
                children.append(traverse_list([node.type], 'type'))
            if node.name:
                # children.append(traverse_list([node.name], 'name'))
                children.append(gen_identifier(node.name, 'name'))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.ClassDef):
            children.append(traverse_list(node.bases, 'bases'))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(
                node.decorator_list, 'decorator_list'))
        elif isinstance(node, ast.AsyncFunctionDef):
            children.append(traverse(node.args))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(
                node.decorator_list, 'decorator_list'))
            if node.returns:
                children.append(traverse_list([node.returns], 'returns'))
        elif isinstance(node, ast.FunctionDef):
            children.append(traverse(node.args))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(
                node.decorator_list, 'decorator_list'))
        # elif isinstance(node, ast.Str):

        else:
            # Default handling: iterate over children.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr_context) or isinstance(child, ast.operator) or isinstance(child,
                                                                                                        ast.boolop) or isinstance(
                    child, ast.unaryop) or isinstance(child, ast.cmpop):
                    # Directly include expr_context, and operators into the type instead of creating a child.
                    json_node['type'] = json_node['type'] + \
                                        type(child).__name__
                else:
                    children.append(traverse(child))

        if isinstance(node, ast.Attribute):
            children.append(gen_identifier(node.attr, 'attr'))

        if (len(children) != 0):
            json_node['children'] = children
        return pos

    traverse(tree)
    return json_tree


def __terminals(ast, node_index, code_tokens, docstring):
    stack, paths = [], []
    list_cursor = 0

    temp_code_tokens = deepcopy(code_tokens)

    def idx_in_list(node):
        node_type = node['type']
        value = node['value']
        if node_type == 'Str':
            for idx in range(list_cursor, len(temp_code_tokens)):
                if "'" in temp_code_tokens[idx] or '"' in temp_code_tokens[idx]:
                    try:
                        source = eval(temp_code_tokens[idx])
                    except SyntaxError:  # 不是这里的锅 而是python2 parse不了
                        continue
                    else:
                        if source == value:
                            return idx
            # print('{} not in code token list'.format(value))
            raise Exception('Str Still not work')
        else:
            try:
                idx = temp_code_tokens.index(value, list_cursor)
            except ValueError:
                # print('{} not in code token list'.format(value))
                raise Exception()
            else:
                return idx

    def dfs(v):
        stack.append(ast[v])
        v_node = ast[v]
        if 'value' in v_node and v_node['type'] == 'Str' and docstring in v_node['value']:
            # that mean this node is comment node -> so ignore 这种清理方式不怎么好 不过其实也没什么无伤大雅
            pass
        elif 'value' in v_node:  # 先检查第一个问题  就是嵌套内部的注释并没有被消除掉
            try:
                idx = idx_in_list(v_node)
            except Exception:
                pass
            else:
                nonlocal list_cursor
                # list_cursor = idx + 1  #

                if v == node_index:  # Top-level func def node.
                    paths.append({'idx': idx, 'path': stack.copy(), 'token': METHOD_NAME})
                    code_tokens[idx] = METHOD_NAME
                else:
                    v_type = v_node['type']
                    if v_type == 'Num':
                        paths.append({'idx': idx, 'path': stack.copy(), 'token': NUM})
                        code_tokens[idx] = NUM
                    elif v_type == 'Str':
                        paths.append({'idx': idx, 'path': stack.copy(), 'token': Str})
                        code_tokens[idx] = Str
                    else:
                        paths.append({'idx': idx, 'path': stack.copy(), 'token': v_node['value']})
                temp_code_tokens[idx] = USED  # 设置一个外部参考的list用来查询 查询到之后将位置上的置空
        if 'children' in v_node:
            for child in v_node['children']:
                dfs(child)
        stack.pop()

    dfs(node_index)
    for i, code in enumerate(code_tokens):
        if "'" in code or '"' in code or '#' in code:
            code_tokens[i] = Str  # 井号形注释并不会出现在ast里边，所以一旦出现在code tokens之后，就无法用ast的方法删除掉
            # ensure that there is not string in code token list
        try:
            _ = float(code)
            code_tokens[i] = NUM
        except ValueError:
            continue
    return paths, code_tokens


def process_path(code_tokens, original_code, docstring, args):
    '''

    :param code_tokens: a list of token
    :param processed_code: code with not comment
    :return:
    paths : path:dict(
    [1,2]:[[a,b,c,d],inter,[e,f,g,h]])
    code_tokens: ['','',]
    '''

    def get_equal_lis(paths, sample_path):
        if len(paths) == 0:
            paths.append(sample_path)
            return len(paths)
        else:
            for i, _lis in enumerate(paths):
                if _lis == sample_path:
                    return i
            paths.append(sample_path)
            return len(paths)

    paths = []
    paths_map = []
    ast = parse_code(original_code)
    func_name = None
    for node_index, node in enumerate(ast):
        if node['type'] == 'FunctionDef':
            func_name = node['value']
            nodes, code_tokens = __terminals(ast, node_index, code_tokens, docstring)
            # there will be more than one function def in source code
            break
    paths = []
    for v_node, u_node in itertools.combinations(
            iterable=nodes,  # TODO：现在有个及其严重的问题，居然出现了（3，3）的情况？？ 首先str型应该不会出现  因为每次找到之后都会直接被干掉了
            r=2,
    ):
        prefix, lca, suffix = __merge_terminals2_paths(v_node['path'], u_node['path'])
        if (len(prefix) + 1 + len(suffix) <= args.max_path_length) \
                and (abs(len(prefix) - len(suffix)) <= args.max_path_width):
            # if len(prefix) + 1 + len(suffix) <= args.max_path_length:
            sample_path = prefix + [lca] + suffix
            paths_map.append([[v_node['idx'], u_node['idx']], get_equal_lis(paths, sample_path)])

            re_sample_path = list(reversed(sample_path))
            paths_map.append([[u_node['idx'], v_node['idx']], get_equal_lis(paths, re_sample_path)])

            # paths[str(tuple((v_node['idx'], u_node['idx'])))] =
            # paths[tuple((v_node['idx'], u_node['idx']))] = prefix + [lca] + suffix

    return paths, code_tokens, func_name, paths_map


def inter_node_map(node_type):
    if node_type in path_dic:
        return path_dic[node_type]
    else:
        length = len(path_dic)
        path_dic[node_type] = length
        return length


def __merge_terminals2_paths(v_path, u_path):
    s, n, m = 0, len(v_path), len(u_path)
    while s < min(n, m) and v_path[s] == u_path[s]:
        s += 1

    prefix = v_path[s:]
    lca = v_path[s - 1]
    suffix = u_path[s:]
    prefix_ = list(reversed([inter_node_map(node['type']) for node in prefix]))
    suffix_ = [inter_node_map(node['type']) for node in suffix]
    lca_ = inter_node_map(lca['type'])
    return prefix_, lca_, suffix_


def __delim_name(name):
    if name in {METHOD_NAME, NUM, Str}:
        return [name]

    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]

    blocks = []
    for underscore_block in name.split('_'):
        blocks.extend(camel_case_split(underscore_block))
    return [block.lower() for block in blocks]  # 这样的分词基本差不多了


def split_token(code_tokens):
    '''

    :param code_tokens: a list of token
    :return: [[sub1,sub2],[sub3,sub4],[],....]
    '''
    processed_tokens = []
    for code in code_tokens:
        processed_tokens.append(__delim_name(code))
    return processed_tokens


length_sqa, path_num = 0, 0
length = 0
c = 0
path_count = dict()
f_length = 0
max_length, max_f_length = 0, 0
code_count = dict()


def count(sample):
    global length_sqa, path_num, length, c, path_count, f_length, max_length, max_f_length, code_count
    c += 1
    s = len(sample['content']) ** 2
    p = len(sample['paths'])
    f = len(sample['target'])
    f_length += f
    length_sqa += s
    path_num += p
    length += len(sample['content'])
    if len(sample['content']) > max_length:
        max_length = len(sample['content'])
    if f > max_f_length:
        max_f_length = f
    for b in sample['paths']:
        _l = len(b)
        # assert _l <= 10
        if _l in path_count:
            path_count[_l] += 1
        else:
            path_count[_l] = 1

    code_len = len(sample['content'])
    if code_len in code_count:
        code_count[code_len] += 1
    else:
        code_count[code_len] = 1

    # if c % 500 == 0:
    #     print('part = {:.5f}'.format(path_num / length_sqa))
    #     print('avg length = {:.5f},max={}'.format(length / c, max_length))
    #     print('avg func length = {:.5f},max={}'.format(f_length / c, max_f_length))


def static_text_vocab(content, func_name):
    for tokens in content:
        for token in tokens:
            if token not in source_dic:
                source_dic[token] = 1
            else:
                source_dic[token] += 1
    for token in func_name:
        if token not in target_dic:
            target_dic[token] = 1
        else:
            target_dic[token] += 1


def convert(text_data, args):
    # all_data = []
    # num = 0
    with open(os.path.join(args.save_dir, args.type + '.json'), 'w') as f:
        for line in tqdm(text_data):
            # if num >= args.file_num:
            #     break
            file_json = json.loads(line)
            original_code = file_json['original_string']
            docstring = file_json['docstring']
            code_tokens = file_json['code_tokens']
            try:
                paths, code_tokens, func_name, paths_map = process_path(code_tokens, original_code, docstring, args)
            except Exception:
                # print('Can not been parsed')  # 明天这个地方设一个统计的东西  然后把所有的东西都统计一下
                continue
            else:
                pass
            content = split_token(code_tokens)
            func_name = split_token([func_name])[0]
            static_text_vocab(content, func_name)
            data = {'target': func_name,
                    'content': content,
                    'paths': paths, 'paths_map': paths_map
                    }  # 这个地方的问题是 第一没有存函数名，第二是应该优化一下路径的字符串 不然还是太大了
            # save_data = [func_name, content, paths]
            # all_data.append(data)
            f.write(json.dumps(data) + '\n')
            # count(data)
            # num += 1
    # return all_data


def process(args):
    dir_path = os.path.join('raw_data', args.type)
    file_list = []
    all_files = os.listdir(dir_path)
    for file in all_files:
        if 'gz' in file:
            continue
        file_list.append(file)
    all_data = []
    print(file_list)
    for file in file_list:
        with open(os.path.join(dir_path, file)) as f:
            sample_file = f.readlines()
            all_data += sample_file

    if args.file:
        with open(os.path.join(dir_path, args.file)) as f:
            sample_file = f.readlines()
            all_data = sample_file
    convert(all_data, args)
    # with open(os.path.join(args.save_dir, args.type + '.pkl', ), 'wb') as f:
    #     pkl.dump(pkl_data, f)
    with open(inter_dic_path, 'wb') as f:
        print('save inter node dic')
        pkl.dump(path_dic, f)

    _keys = sorted(path_count.keys())
    _c = 0
    _s = 0
    for k in _keys:
        _s += path_count[k]

    for k in _keys:
        _c += path_count[k]
        print('len of path <={} has {}'.format(k, _c / _s))

    _keys = sorted(code_count.keys())
    _c = 0
    _s = 0
    for k in _keys:
        _s += code_count[k]

    for k in _keys:
        _c += code_count[k]
        print('len of code <={} has {}'.format(k, _c / _s))

    if args.text_vocab:
        print('Save Text Vocab')
        with open('./data/source_vocab.json', 'w') as f:
            json.dump(source_dic, f)
        with open('./data/target_vocab.json', 'w') as f:
            json.dump(target_dic, f)
    print('Inter Node Vocab Size = {}'.format(len(path_dic)))
    print('Source Vocab Size = {}'.format(len(source_dic)))
    print('Target Vocab Size = {}'.format(len(target_dic)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['train', 'valid', 'test'], type=str, default='train')
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--file', type=str, )  # default='./test.jsonl' not work
    parser.add_argument('--file_num', type=int, default=1000)  # not work
    parser.add_argument('--max_path_length', type=int, default=8)
    parser.add_argument('--max_path_width', type=int, default=2)
    parser.add_argument('--text_vocab', type=boolean_string, default=False)
    args = parser.parse_args()
    process(args)

    # 首先执行train文件 保存文字词表和节点词表 text_vocab=True
    # 然后执行test和valid文件  更新节点词表 text_vocab=False
    # 最后得到最终节点词表大小

    # 保存文字词表时 要分别保存source端和target端  建立{str:count}的映射后 直接存储为json文件
    # 节点词表的格式是 {str:idx}

    # 统计一下 函数名的长度范围

# type=test text_vocab=True
# Inter Node Vocab Size = 125
# Source Vocab Size = 17448
# Target Vocab Size = 6290

# type=valid text_vocab=False
# Inter Node Vocab Size = 132
# Source Vocab Size = 21363
# Target Vocab Size = 7675
'''
这个文件搞完之后是这样的：
每个样本：
content:[[sub1,sub2],[sub3,sub4],[],....]
path:dict(
    [1,2]:[[a,b,c,d],[e,f,g,h]]
)
'''
'''
随后在dataset里边每个样本进行词典映射
'''
'''

tranformer那篇估计有

'''
'''
为啥ast解析完之后  ’v‘ 变成了 v
r也没有了  导致我根本就查不到
所以应该将ast解析之后的str 加一个卷边  然后再将code tokens里边的全部干掉


问题是这样的：现在获得了从当前节点到最顶端funcdef的路径
但是两个断点的路径之间是怎么处理的？？ 因为并不是直接拼起来就可以的

第二个问题是 有些内部节点的value也会出现在源码里 比方说func和keyword 这个的确是可以加进去

但是核心还是要解决这个路径的问题  


回头研究一下 这样搞到底对不对

然后第三个问题是  有些叶子节点的并不会出现在里边  比方说有一个 keyword是None 这个就很恶心了 
这个等会解决

'''

# TODO： 2）json文件太大的问题必须解决，可以先从train集合开始 处理的同时把中间节点统计一下 然后这样存储的话应该就会小很多 3）进行训练集的统计任务 并构建词表
# 1000个样本 137mb  不存储path为2.46mb
# path太多了 这不行啊
# 使用限制之后8 变为14.8mb  part从0.08降低为0.01 统计一下路径长度
# 仅仅使用path长度为16的限制 大小为84.9mb
# 不使用词典 使用list进行存储 大小为84.8mb 不用str用元组 大小为70.3mb

# 仅仅使用path长度为20的限制 大小为97.8mb part等于0.08

# java small 训练集5.01G 测试284mb
# 总共700K个样本  所以1K个样本7.3MB
# 所以先用8吧 不然还是太大了

'''

现在选取了前2000个样本，按照path长度进行百分比的统计
len of path <=1 has 0.0003228411838744987
len of path <=2 has 0.00035005964434166954
len of path <=3 has 0.010018661656957804
len of path <=4 has 0.021828071192151556
len of path <=5 has 0.04495903716208234
len of path <=6 has 0.07339098732508123
len of path <=7 has 0.1334865127802958
len of path <=8 has 0.2075175119605287
len of path <=9 has 0.3100460502328029
len of path <=10 has 0.41656965364698073
len of path <=11 has 0.528671529320236
len of path <=12 has 0.6253422390623845
len of path <=13 has 0.7116328864780389
len of path <=14 has 0.777801530924991
len of path <=15 has 0.8321070849841613
len of path <=16 has 0.871400713010254
len of path <=17 has 0.901671232374393
len of path <=18 has 0.9230991543942598
len of path <=19 has 0.9394572601179428
len of path <=20 has 0.9508943059925788
len of path <=21 has 0.9596076156496319
len of path <=22 has 0.9656986912645943
len of path <=23 has 0.9704657021880807
len of path <=24 has 0.9737993965440493
len of path <=25 has 0.9764968593865979
len of path <=26 has 0.9783312702118333
len of path <=27 has 0.9798997339962541
len of path <=28 has 0.98111776010216
len of path <=29 has 0.982159433266289
len of path <=30 has 0.9829882731909316
len of path <=31 has 0.9836634422241868
len of path <=32 has 0.98430005177178
len of path <=33 has 0.9850060305901472
len of path <=34 has 0.985739983937328
len of path <=35 has 0.9864665656181322
len of path <=36 has 0.9869644366241775
len of path <=37 has 0.9874326319476302
len of path <=38 has 0.9878588654778626
len of path <=39 has 0.9882196990961392
len of path <=40 has 0.9885601188690654
len of path <=41 has 0.9888801247966412
len of path <=42 has 0.9891821741009922
len of path <=43 has 0.9894730713972351
len of path <=44 has 0.9897658588643438
len of path <=45 has 0.9900788711597163
len of path <=46 has 0.9903627748237558
len of path <=47 has 0.990713023485184
len of path <=48 has 0.99108935650456
len of path <=49 has 0.9914471658494513
len of path <=50 has 0.9917788908363949
len of path <=51 has 0.9921125059942043
len of path <=52 has 0.9924107749568237
len of path <=53 has 0.9927375854995164
len of path <=54 has 0.9930458723677245
len of path <=55 has 0.9933447083816036
len of path <=56 has 0.9936480808055606
len of path <=57 has 0.9939495630586518
len of path <=58 has 0.99423630197899
len of path <=59 has 0.9945379732491678
len of path <=60 has 0.994811291956359
len of path <=61 has 0.9950687332282776
len of path <=62 has 0.9953063277061056
len of path <=63 has 0.9955418429959814
len of path <=64 has 0.9957410670052341
len of path <=65 has 0.9960958520767402
len of path <=66 has 0.9963738962110958
len of path <=67 has 0.9965918329119198
len of path <=68 has 0.9968564568331283
len of path <=69 has 0.9971044472507181
len of path <=70 has 0.9972460210485647
len of path <=71 has 0.9973972347178268
len of path <=72 has 0.997562246634409
len of path <=73 has 0.9978722346563962
len of path <=74 has 0.9980990551602893
len of path <=75 has 0.9983506369025241
len of path <=76 has 0.9985186730924915
len of path <=77 has 0.9986337844982173
len of path <=78 has 0.9987420912888262
len of path <=79 has 0.9988515321819547
len of path <=80 has 0.9989564366650052
len of path <=81 has 0.9990785417029343
len of path <=82 has 0.9991862814422835
len of path <=83 has 0.9993029049847019
len of path <=84 has 0.9994064863481464
len of path <=85 has 0.9995181954463137
len of path <=86 has 0.9996151612117281
len of path <=87 has 0.9997208217631249
len of path <=88 has 0.9998115499646821
len of path <=89 has 0.9998960406023824
len of path <=90 has 0.9999531237625288
len of path <=91 has 0.9999710803857537
len of path <=92 has 0.9999858237185066
len of path <=93 has 0.9999965976924416
len of path <=94 has 1.0
avg length = 129.68 part=0.08
part计算方式是路径个数除以length的平方

所以考虑将path的长度设置为20，

'''

'''
仅处理valid数据
Inter Node Vocab Size = 128
Source Vocab Size = 21363
Target Vocab Size = 7675

'''

'''
这里边会存在很多完全一样的边  所以可以先建立一个边的不重复的list
然后这样就可以节省很多空间

注意：： 在计算的时候 设置的max path num应该是unique的path的num
因为相同的path batch是没有什么意义的 纯属浪费计算资源

'''
