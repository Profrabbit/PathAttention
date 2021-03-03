from tree_sitter import Language, Parser
import argparse
import os
import json
import itertools
import string
import re
from tqdm import tqdm
import attr
import random
from multiprocessing import Process


@attr.s
class MyNode:
    type = attr.ib()
    named = attr.ib()
    idx = attr.ib()


def init_parser(args):
    Language.build_library(
        'build/{}.so'.format(args.language),
        [
            'vendor/tree-sitter-{}'.format(args.language),
        ]
    )
    language = Language('build/{}.so'.format(args.language), args.language)
    lang_parser = Parser()
    lang_parser.set_language(language)
    return lang_parser


def read_files(args):
    dir_path = os.path.join('raw_data', args.language, args.type)
    file_list = []
    all_files = os.listdir(dir_path)
    for file in all_files:
        if 'gz' in file:
            continue
        file_list.append(file)
    all_data = []
    for file in tqdm(file_list):
        with open(os.path.join(dir_path, file)) as f:
            lines = f.readlines()
            for line in lines:
                all_data.append(json.loads(line)['code'])
    print('Load {} {} files => {}'.format(args.language, args.type, len(all_data)))
    return all_data


def language_parse(args, data, lang_parser):
    tree = lang_parser.parse(bytes(data, "utf-8"))
    stack, paths = [], []
    paths_map = dict()
    path_pool = []

    def dfs(node):
        stack.append(node)
        if node.type == 'string' or node.child_count == 0:
            paths.append(stack.copy())
        else:
            for child in node.children:
                dfs(child)
        stack.pop()

    def split_word(word):
        def camel_case_split(identifier):
            matches = re.finditer(
                '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
                identifier,
            )
            return [m.group(0) for m in matches]

        blocks = []
        for underscore_block in word.split('_'):
            blocks.extend(camel_case_split(underscore_block))
        return [block.lower() for block in blocks]

    def clean_convert_split(args, paths, data):
        data_lines = data.splitlines()
        temp_paths = []
        func_name = []
        count = 0
        get_func = False
        for idx, path in enumerate(paths):
            if count >= args.max_code_length:
                break
            terminal = path[-1]
            if terminal.type in string.punctuation and not args.punctuation:
                continue
            if terminal.type == 'identifier':
                l_, r_ = terminal.start_point, terminal.end_point
                assert l_[0] == r_[0]  # assert at same line
                literal = data_lines[l_[0]][l_[1]: r_[1]]
                blocks = split_word(literal)
                if not get_func:  # this is func name
                    func_name = blocks
                    new_node = MyNode('<METHOD>', terminal.is_named, count)
                    count += 1
                    temp_paths.append(path[:-1] + [new_node])
                    get_func = True
                else:
                    for block in blocks:
                        new_node = MyNode(block, terminal.is_named, count)
                        count += 1
                        temp_paths.append(path[:-1] + [new_node])
            elif terminal.type in ['integer', 'float', 'string']:
                new_node = MyNode('<{}>'.format(terminal.type), terminal.is_named, count)
                count += 1
                temp_paths.append(path[:-1] + [new_node])
            else:
                new_node = MyNode(terminal.type, terminal.is_named, count)
                count += 1
                temp_paths.append(path[:-1] + [new_node])
        return temp_paths, func_name

    def merge_terminals2_paths(v_path, u_path):
        s, n, m = 0, len(v_path), len(u_path)
        while s < min(n, m) and v_path[s] is u_path[s]:
            s += 1
        prefix, l_node = v_path[s:-1], v_path[-1]
        lca = v_path[s - 1]
        suffix, r_node = u_path[s:-1], u_path[-1]
        prefix = list(reversed(prefix))
        path = prefix + [lca] + suffix
        return l_node, prefix, [node.type for node in path], suffix, r_node

    def save_path(path, path_pool):
        if len(path_pool) == 0:
            path_pool.append(path)
        else:
            for i, _lis in enumerate(path_pool):
                if _lis == path:
                    return i
            path_pool.append(path)
        return len(path_pool) - 1

    cursor = tree.walk()
    dfs(cursor.node)
    paths, func_name = clean_convert_split(args, paths, data)
    terminals = [path[-1] for path in paths]
    combinations = []
    if args.path_radius > 0:
        for i in range(len(paths)):
            for j in range(i + 1, min(len(paths), args.path_radius + i + 1)):
                combinations.append(tuple((paths[i], paths[j])))
    else:
        combinations = itertools.combinations(iterable=paths, r=2, )
    for v_path, u_path in combinations:
        if abs(len(v_path) - len(u_path)) > args.max_path_length + 1: continue
        l_node, prefix, path, suffix, r_node = merge_terminals2_paths(v_path, u_path)
        if len(path) <= args.max_path_length and (abs(len(prefix) - len(suffix)) <= args.max_path_width):
            path_idx = save_path(path, path_pool)
            re_path = list(reversed(path))
            re_path_idx = save_path(re_path, path_pool)
            if path_idx in paths_map:
                paths_map[path_idx].append(terminals.index(l_node))
                paths_map[path_idx].append(terminals.index(r_node))
            else:
                paths_map[path_idx] = [terminals.index(l_node), terminals.index(r_node)]
            if re_path_idx in paths_map:
                paths_map[re_path_idx].append(terminals.index(r_node))
                paths_map[re_path_idx].append(terminals.index(l_node))
            else:
                paths_map[re_path_idx] = [terminals.index(r_node), terminals.index(l_node)]  # 改了一下path map的接口
    return path_pool, [node.type for node in terminals], [int(node.named) for node in terminals], func_name, paths_map


def statistic(source_dic, target_dic, code_tokens, func_name):
    def lookup_update(dict, item):
        if item in dict:
            dict[item] += 1
        else:
            dict[item] = 1
        return list(dict.keys()).index(item)

    for token in code_tokens:
        _ = lookup_update(source_dic, token)
    for token in func_name:
        _ = lookup_update(target_dic, token)


def data_count(data, count_dic):
    paths, code_tokens, code_named, func_name, paths_map = \
        data['paths'], data['content'], data['named'], data['target'], data['paths_map'],
    count_dic['tokens'] += len(code_tokens)
    count_dic['uni_paths'] += len(paths)
    count_dic['paths'] += (sum([len(val) for key, val in paths_map.items()]) / 2)
    count_dic['named'] += code_named.count(1)
    count_dic['func'] += len(func_name)
    count_dic['nums'] += 1


def sub_process(args, idx, all_data, lang_parser):
    save_path = os.path.join('data', args.language, '{}_{}.json'.format(args.type, idx))
    source_dic, target_dic = dict(), dict()
    with open(save_path, 'w') as f:
        for data in tqdm(all_data):
            paths, code_tokens, code_named, func_name, paths_map = language_parse(args, data, lang_parser)
            statistic(source_dic, target_dic, code_tokens, func_name)
            data = {'target': func_name,
                    'content': code_tokens, 'named': code_named,
                    'paths': paths, 'paths_map': paths_map
                    }
            f.write(json.dumps(data) + '\n')
    dict_save_path = os.path.join('data', args.language, '{}_dict_{}.json'.format(args.type, idx))
    with open(dict_save_path, 'w') as f:
        f.write(json.dumps(source_dic) + '\n')
        f.write(json.dumps(target_dic) + '\n')


def dict_init(args, node_dic, count_dic):
    node_dic_path = os.path.join('data', args.language, 'node_vocab.json')
    if os.path.exists(node_dic_path):
        with open(node_dic_path, 'r') as f:
            print('Already exist inter node dic')
            saved_node_dic = json.loads(f.readline())
            for key, value in saved_node_dic.items():
                node_dic[key] = value
    print(node_dic)
    keys = ['tokens', 'uni_paths', 'paths', 'named', 'func', 'nums']
    for k in keys:
        count_dic[k] = 0


def update_sum_dict(sub_source_dic, sub_target_dic, source_dic, target_dic):
    def update_dict(sub_dic, dic):
        for key, value in sub_dic.items():
            if key in dic:
                dic[key] += value
            else:
                dic[key] = value

    update_dict(sub_source_dic, source_dic)
    update_dict(sub_target_dic, target_dic)


def path_convert(data, node_dic):
    def lookup(dic, item):
        if item not in dic:
            dic[item] = len(dic)
        return dic[item]

    path = data['paths']
    temp_path = []
    for p in path:
        temp_p = []
        for node in p:
            temp_p.append(lookup(node_dic, node))
        temp_path.append(temp_p)
    data['paths'] = temp_path


def process(args):
    lang_parser = init_parser(args)
    if not os.path.exists('data/{}'.format(args.language)): os.makedirs('data/{}'.format(args.language))
    node_dic, source_dic, target_dic, count_dic = dict(), dict(), dict(), dict()
    dict_init(args, node_dic, count_dic)
    all_data = read_files(args)
    random.shuffle(all_data)
    if args.nums > 0: all_data = all_data[:args.nums]
    pool = []
    split_data = [[] for _ in range(args.process_num)]
    for i in range(len(all_data)):
        split_data[i % args.process_num].append(all_data[i])
    for i in range(args.process_num):
        pool.append(Process(target=sub_process, args=(args, i, split_data[i], lang_parser)))
        pool[-1].start()
    for p in pool:
        p.join()

    print('Sub Files Merge')
    sum_save_path = os.path.join('data', args.language, '{}.json'.format(args.type))
    with open(sum_save_path, 'w') as f:
        for i in range(args.process_num):
            sub_save_path = os.path.join('data', args.language, '{}_{}.json'.format(args.type, i))
            with open(sub_save_path, 'r') as l:
                lines = l.readlines()
                for line in lines:
                    data = json.loads(line)
                    data_count(data, count_dic)
                    path_convert(data, node_dic)
                    f.write(json.dumps(data) + '\n')
            os.remove(sub_save_path)

    print('Sub Dict Concat')
    for i in range(args.process_num):
        sub_dict_path = os.path.join('data', args.language, '{}_dict_{}.json'.format(args.type, i))
        with open(sub_dict_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            sub_source_dic = json.loads(lines[0])
            sub_target_dic = json.loads(lines[1])
            update_sum_dict(sub_source_dic, sub_target_dic, source_dic, target_dic)
        os.remove(sub_dict_path)

    print('avg_tokens:{}'.format(count_dic['tokens'] / count_dic['nums']))
    print('avg_uni_paths:{}'.format(count_dic['uni_paths'] / count_dic['nums']))
    print('avg_paths:{}'.format(count_dic['paths'] / count_dic['nums']))
    print('named_ratio:{}'.format(count_dic['named'] / count_dic['tokens']))
    print('path_ratio:{}'.format((count_dic['paths'] / count_dic['nums']) / (
            (count_dic['tokens'] / count_dic['nums']) * (count_dic['tokens'] / count_dic['nums'] - 1) / 2)))
    print('source_vocab:{}'.format(len(source_dic)))
    print('target_vocab:{}'.format(len(target_dic)))
    print('node_vocab:{}'.format(len(node_dic)))
    if args.type == 'train':
        print('Save Text Vocab')
        with open('./data/{}/source_vocab.json'.format(args.language), 'w') as f:
            json.dump(source_dic, f)
        with open('./data/{}/target_vocab.json'.format(args.language), 'w') as f:
            json.dump(target_dic, f)
    print('Save Node Vocab')
    with open('./data/{}/node_vocab.json'.format(args.language), 'w') as f:
        json.dump(node_dic, f)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['python', 'javascript'], type=str, default='python')
    parser.add_argument('--type', choices=['train', 'valid', 'test'], type=str, default='valid')
    parser.add_argument('--max_path_length', type=int, default=8)
    parser.add_argument('--max_path_width', type=int, default=2)
    parser.add_argument('--max_code_length', type=int, default=512)
    parser.add_argument('--path_radius', type=int, default=-1)
    parser.add_argument('--nums', type=int, default=-1)
    parser.add_argument('--punctuation', type=boolean_string, default=False)
    parser.add_argument('--process_num', type=int, default=64)
    args = parser.parse_args()
    print(args)
    process(args)

    '''


valid 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 361/361 [13:26<00:00,  2.23s/it]
Sub Files Merge
Sub Dict Concat
avg_tokens:75.20569524386549
avg_uni_paths:231.12273337084
avg_paths:2529.261089713074
named_ratio:0.8651910658323445
path_ratio:0.9064329611467545
source_vocab:20697
target_vocab:7736
node_vocab:79
Save Node Vocab
664657kb / 23104 28.76



train
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6440/6440 [3:41:34<00:00,  2.06s/it]
Sub Files Merge
Sub Dict Concat
avg_tokens:73.34297560762583
avg_uni_paths:225.00047552271107
avg_paths:2420.3138595461182
named_ratio:0.8645519317483507
path_ratio:0.9123191466778391
source_vocab:125590
target_vocab:44045
node_vocab:81
Save Text Vocab
Save Node Vocab

test
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 347/347 [12:12<00:00,  2.11s/it]
Sub Files Merge
Sub Dict Concat
avg_tokens:73.7535624098124
avg_uni_paths:224.61183261183263
avg_paths:2448.6152597402597
named_ratio:0.868050617556444
path_ratio:0.9126689623989318
source_vocab:17009
target_vocab:6348
node_vocab:81
Save Node Vocab


'''
