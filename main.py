import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import fairies as fa 

# keras_version == 0.10.0

maxlen = 256

train_data = fa.read_json('train_data.json')

table = fa.read_excel('schema.xlsx','工作表1')
rowNum = table.nrows
colNum = table.ncols

o_list = []
for i in range(1,rowNum):
    o_list.append(table.cell(i,1).value + 's_' + table.cell(i,2).value)
    o_list.append(table.cell(i,1).value + 'o_' + table.cell(i,4).value)

o_list = list(set(o_list))
o_list.insert(0,'i')
o_list.insert(1,'o')

print(o_list)
print(len(o_list))

o_list = ['i', 'o', '首都o_城市', '获奖o_作品', '邮政编码s_行政区', '所在城市o_城市', '毕业院校o_学校', '票房s_影视作品', '父亲s_人物', '专业代码s_学科专业', '国籍s_人物', '编剧s_影视作品', '成立日期o_Date', '专业代码o_Text', '主演s_影视作品', '获奖s_娱乐人物', '歌手s_歌曲', '主角s_文学作品', '官方语言o_语言', '校长o_人物', '所在城市s_景点', '主角o_人物', '简称o_Text', '主题曲o_歌曲', '获奖o_奖项', '母亲o_人物', '主演o_人物', '出品公司s_影视作品', '主持人o_人物', '上映时间o_Date', '校长s_学校', '作者s_图书作品', '朝代o_Text', '上映时间o_地点', '配音s_娱乐人物', '国籍o_国家', '妻子o_人物', '气候s_行政区', '朝代s_历史人物', '配音o_人物', '面积o_Number', '注册资本o_Number', '作词s_歌曲', '嘉宾o_人物', '简称s_机构', '作词o_人物', '总部地点o_地点', '丈夫s_人物', '出品公司o_企业', '面积s_行政区', '代言人s_企业/品牌', '票房o_Number', '创始人o_人物', '邮政编码o_Text', '所属专辑o_音乐专辑', '饰演o_人物', '作曲s_歌曲', '号o_Text', '成立日期s_机构', '获奖o_Date', '编剧o_人物', '所属专辑s_歌曲', '修业年限o_Number', '创始人s_企业', '人口数量o_Number', '占地面积s_机构', '嘉宾s_电视综艺', '导演s_影视作品', '丈夫o_人物', '主题曲s_影视作品', '歌手o_人物', '占地面积o_Number', '作者o_人物', '注册资本s_企业', '制片人o_人物', '祖籍o_地点', '制片人s_影视作品', '改编自s_影视作品', '人口数量s_行政区', '导演o_人物', '号s_历史人物', '饰演s_娱乐人物', '改编自o_作品', '代言人o_人物', '父亲o_人物', '海拔s_地点', '母亲s_人物', '毕业院校s_人物', '妻子s_人物', '作曲o_人物', '上映时间s_影视作品', '获奖o_Number', '票房o_地点', '官方语言s_国家', '主持人s_电视综艺', '祖籍s_人物', '首都s_国家', '董事长s_企业', '饰演o_影视作品', '修业年限s_学科专业', '配音o_影视作品', '董事长o_人物', '气候o_气候', '海拔o_Number', '总部地点s_企业']

id2label,label2id = fa.label2id(o_list)
num_labels = len(o_list) 

batch_size = 32

def read_data(filename):
    train_data = fa.read_json(filename)
    res = []
    for i in train_data:
        predicts = []
        text = i['text']
        for spo in i['spo_list']:
            # print(spo)
            schema_type = spo['predicate'] + '_' + spo['object_type']['@value']
            new = {}
            new['predicate'] = spo['predicate']
            new['o_value'] = spo['object']['@value']
            new['o_type'] = spo['object_type']['@value']
            new['s_value'] = spo['subject']
            new['s_type'] = spo['subject_type']
            predicts.append(new)
        res.append([text,predicts])
    return res        

a = read_data('train_data.json')

config_path = 'D:/lyh/model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:/lyh/model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'D:/lyh/model/chinese_L-12_H-768_A-12/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def search(pattern, sequence):

    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """

    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class data_generator(DataGenerator):

    """数据生成器

    """

    def __iter__(self, random=False):

        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, result in self.sample(random):

            text = result[0]
            predicts = result[1]

            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)

            seq_len = len(token_ids)

            labels = [[0] * num_labels for i in range(seq_len)]

            for predict in predicts:
                
                p = predict['predicate']

                o_type = p + 'o_' + predict['o_type']
                o_value = predict['o_value']
                s_type = p + 's_' + predict['s_type']
                s_value = predict['s_value']
                
                o_token_ids = tokenizer.encode(o_value)[0][1:-1]
                o_start = search(o_token_ids, token_ids)
              
                o_type_index = label2id[o_type]
                labels[o_start][o_type_index] = 1 

                for i in range(1,len(o_token_ids)):
                    labels[o_start + i][1] = 1
                
                s_token_ids = tokenizer.encode(s_value)[0][1:-1]
                s_start = search(s_token_ids, token_ids)

                s_type_index = label2id[s_type]
                labels[s_start][s_type_index] = 1
                for i in range(1,len(s_token_ids)):
                    labels[s_start + i][1] = 1
                
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,

)

output = Dense(units=num_labels,
               activation='sigmoid',
               kernel_initializer=model.initializer)(model.output)
model = Model(model.input, output)
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),
    metrics=['accuracy']
)

train_generator = data_generator(a, 8)

def extract_arguments(text):
    
    # 等你真的到了这里 你才能懂这里的风景
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 510:
        tokens.pop(-2)

    mapping = tokenizer.rematch(text, tokens)
    
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    labels = model.predict([[token_ids], [segment_ids]])[0]
    labels = labels[1:]
    for lable in labels:
        for i in range(len(lable)):
            if lable[i] >= 0.5:
                lable[i] = 1
            else:
                lable[i] = 0 

    s_entries, o_entries = find_entry(labels,mapping,text)
    return s_entries, o_entries

def find_entry(labels,mapping,text):

    # 分开o_res和s_res
    o_res = []
    s_res = []


    for k,label in enumerate(labels):
        for i,l in enumerate(label):
            if l == 1 and i != 1:
                start_type = id2label[i]
                start = k
                end = 0
                j = k + 1
                while j < len(labels) and labels[j][1] == 1:
                    end = j
                    j += 1
                if end > start:
                    entry = text[mapping[start+1][0]:mapping[end+1][-1] +1]
                    if 's_' in start_type:
                        s_res.append([entry,start_type])
                    if  'o_' in start_type:
                        o_res.append([entry,start_type])

    return s_res,o_res

def find_relation(s_entries, o_entries):
    
    res = []

    for s_ in s_entries:
        s_type = s_[1][:s_[1].find('s_')]
        o_type = s_type + 'o_'
        for o_ in o_entries:
            if o_type in o_[1]:
                res.append([s_[0],o_[0],s_type])

    return res            

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('best_model.weights')


evaluator = Evaluator()
model.summary()

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=20,
    callbacks=[evaluator]
)


# model.load_weights('best_model.weights')

# text_cases = fa.read_txt('text.txt')

# final_res = []

# for text_case in text_cases:

#     s_entries, o_entries = extract_arguments(text_case)

#     res = find_relation(s_entries, o_entries)

#     final_res.append([text_case,res])

# fa.write_json('final_res.json',final_res)
        