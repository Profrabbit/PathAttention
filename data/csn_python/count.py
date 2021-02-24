import json

with open('./data/target_vocab.json') as f:
    data = f.readlines()
    vocab = json.loads(data[0])
    ordered_list = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
    sum_ = 0
    _vocab = dict()
    count = 0
    for key, value in ordered_list:
        sum_ += value
    for key, value in ordered_list:
        count += value
        if count / sum_ >= 0.999:
            break
        _vocab[key] = len(_vocab)
    print(len(vocab))
    print(len(_vocab))
    # print(_vocab)

with open('./data/source_vocab.json') as f:
    data = f.readlines()
    vocab = json.loads(data[0])
    ordered_list = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
    sum_ = 0
    _vocab = dict()
    count = 0
    a = 0
    for key, value in ordered_list:

        a += 1
        sum_ += value

    for key, value in ordered_list:

        count += value
        if count / sum_ >= 0.999:
            break
        _vocab[key] = len(_vocab)
    print(len(vocab))
    print(len(_vocab))
    # print(_vocab)
'''

43568
34480
126883
126890
97152


43568
34480
126890
126890
91473

'''