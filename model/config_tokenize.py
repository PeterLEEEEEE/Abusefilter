import tensorflow as tf
import pandas as pd
import numpy as np
import re
import json
import numpy as np
from konlpy.tag import Okt
import tensorflow as tf
import pickle
import pandas as pd

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
import re
import os


def preprocess(comment, okt, _stem=False):

    try:
        comment_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", comment)
        comment_text = re.sub("[\n]", "", comment_text)
        word_comment = okt.morphs(comment_text, stem=_stem)  # 어간 추출(해야지 -> 하다)
        # if word_comment:
        #   return word_comment.lower()
    except TypeError:
        print(train_data[train_data['sentence'] == comment])
    except:
        print(comment)

    return word_comment


def make_configs(word_vocab, id_to_label):

    data_configs = {}
    data_configs['vocab'] = word_vocab
    data_configs['vocab_size'] = len(word_vocab) + 1
    data_configs['label'] = id_to_label
    data_configs['sentence_len'] = 22
    # print(data_configs)
    save_configs = json.dumps(data_configs)

    with open("/content/drive/My Drive/Abuse_filter/configs/data_configs2.json", 'w') as f:
        f.write(save_configs)


def make_tokenizer():
    train_data = pd.read_csv(
        '/content/drive/My Drive/Abuse_filter/data/hate_speech_binary_dataset_ver1.0.csv')
    okt = Okt()
    label_to_id = {'욕설': 0, '정상': 1}
    id_to_label = {0: '욕설', 1: '정상'}

    inputs, outputs = [], []

    for sentence, label in zip(train_data['sentence'], train_data['label']):
        inputs.append(preprocess(sentence, okt, True))
        outputs.append(label)

    tokenizer = Tokenizer(oov_token="<UNK>",)  # 없는 단어 1으로 표시
    # fit_on_texts() 메서드는 문자 데이터를 입력받아서 리스트의 형태로 변환
    tokenizer.fit_on_texts(inputs)
    # tokenizer의 word_index 속성은 단어와 숫자의 키-값 쌍을 포함하는 딕셔너리를 반환
    word_vocab = tokenizer.word_index

    make_configs(word_vocab, id_to_label)  # data_configs2 생성
    inputs = tokenizer.texts_to_sequences(inputs)
    padded_inputs = pad_sequences(inputs, maxlen=22, padding='post')
    padded_inputs = padded_inputs.tolist()
    output = outputs

    model_train = {
        'inputs': padded_inputs,
        'outputs': outputs
    }

    with open('/content/drive/My Drive/Abuse_filter/data/model_train.pickle', 'wb') as fp:
        pickle.dump(model_train, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('/content/drive/My Drive/Abuse_filter/configs/tokenizer2.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_tokenizer():
    if os.path.isfile('/content/drive/My Drive/Abuse_filter/tokenizer2.pickle'):
        with open('/content/drive/My Drive/Abuse_filter/tokenizer2.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    else:
        make_tokenizer()
        try:
            with open('/content/drive/My Drive/Abuse_filter/tokenizer2.pickle', 'rb') as f:
                tokenizer = pickle.load(f)
        except:
            print('No tokenizer exist!')
