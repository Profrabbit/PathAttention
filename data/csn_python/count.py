import json


def count(type):
    a, b, c = 0, 0, 0
    d = 0
    uni_dic = dict()

    path = './data/{}.json'.format(type)
    with open(path, 'r') as f:
        for line in f.readlines():
            d += 1
            data = json.loads(line)
            uni_path_num = len(data['paths'])
            if uni_path_num in uni_dic:
                uni_dic[uni_path_num] += 1
            else:
                uni_dic[uni_path_num] = 1
            path_num = len(data['paths_map'])
            num = len(data['content'])
            a += uni_path_num
            b += path_num
            c += (num * num)
    print(a / c)
    print(b / c)
    print(a / d)
    print(b / d)
    ordered_list = sorted(uni_dic.items(), key=lambda item: item[0], reverse=False)
    temp = 0
    for k, v in ordered_list:
        temp += v
        print('<={} => {}'.format(k, temp / d))


if __name__ == '__main__':
    count(type='valid')
    '''
    <=512 => 0.920781118540495
    百分之92的样本的独特的边小于512条
    
    '''