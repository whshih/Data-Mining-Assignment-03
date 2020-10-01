
import json
import time
import re
import math
from collections import Counter
import sys

def scc():
    from pyspark import SparkContext
    from pyspark import SparkConf
    confSpark = SparkConf().setAppName('553').setMaster(
        "local[*]").set('spark.driver.memory', '4G').set(
            'spark.executor.memory', '4G')
    sc = SparkContext.getOrCreate(confSpark)
    return sc

def tojson(line):
    lines = line.splitlines()
    data = json.loads(lines[0])
    return data

def list2dic(list):
    dic = {}
# Break the collection of user_id and count in to user_id dictionary --> dictionary={ row num: 'user_id'}
# The num of user_id = 26184
    for i in range(len(list)):
        if list[i][0] in dic:
            break
        else:
            dic[list[i][0]] = i
    return dic

def sanitize(data, stopwords):
    words = re.findall("[a-zA-Z]+", data)
    ret = []
    for word in words:
        word = word.lower()
        if word in stopwords:
            continue
        else:
            ret.append(word)
    return ret

def tf_count(bus_text: list):
    # Total number of [('business_id', ['case','eat',...]),...]
    dic = {}
    tf = {}
    for i in range(len(bus_text)):
        if bus_text[i] in dic:
            dic[bus_text[i]] += 1
        else:
            dic[bus_text[i]] = 1
    max_values = max(dic.values())
    for i in range(len(bus_text)):
        tf[bus_text[i]] = dic[bus_text[i]]/max_values
    return tf

def text_count(bus_text: list):
    # Total number of [('business_id', ['case','eat',...]),...]
    dic = {}
    for i in range(len(bus_text)):
        if bus_text[i] in dic:
            dic[bus_text[i]] += 1
        else:
            dic[bus_text[i]] = 1
    return dic

def idf_(times, bus_num):
    #  [('kapua', 3), ('sonnie', 1), ('ashman', 6), ('talus', 1), ('reimbursements', 2), ....]
    return math.log2(bus_num/times)

def tfidf_(tf: dict, idf: dict):
    #  tf  = {'m': 0.36363636363636365, 'officially': 0.045454545454545456,...}
    #  idf = {'gello': 9.235325624670091, 'donday': 9.235325624670091, 'sabrinacosta': 9.235325624670091,...}
    dic = {}
    for i in tf.keys():
        dic[i] = tf[i] * idf[i]

    top200 = sorted(dic, key=lambda key: -dic[key])
    list1 = []
    for i in top200:
        if i in list1:
            break
        else:
            list1.append(i)
    return list1[:200]


def save_model(data1, data2, file_path):
    import pickle
    with open(file_path, 'wb') as model:
        pickle.dump(data1, model)
        pickle.dump(data2, model)


def main():
    time_start = time.time()
    sc = scc()

    data = 'train_review.json' #sys.argv[1]
    output_file_name = 'task2.model' #sys.argv[2]
    stopwords_file = 'stopwords' #sys.argv[3]
    textRDD = sc.textFile(data).map(tojson)
    with open(stopwords_file, "r") as fd:
        stopwords = fd.read().splitlines()

    # Total number of [('business_id', ['case','eat',...]),...]
    bus_text = textRDD.map(lambda key: (key['business_id'], key['text'])).map(
        lambda x: (x[0], sanitize(x[1], stopwords))).reduceByKey(lambda a, b: a + b)
    bus_num = textRDD.map(lambda key: key['business_id']).distinct().count()

    # [('business', {'m': 0.36363636363636365, 'officially': 0.045454545454545456,...}),(),....]
    tf = bus_text.map(lambda x: (x[0], tf_count(x[1])))
    df = bus_text.mapValues(lambda x: set(x)).flatMap(lambda x: (x[1])).map(
        lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

    # dict = {'hello': 9.235325624670091, 'donday': 9.235325624670091, 'sabrinacosta': 9.235325624670091,...}
    idf = df.map(lambda word: (word[0], idf_(word[1], bus_num))).collectAsMap()
    #        business_id              word k         tf-idf
    # business_profile_dic = {'INUsvJhd-im_nS0VEIKH7Q': ['located','next',...],....}
    # 634 businesses have the vector whose length less than 200
    business_profile = tf.map(lambda x: (
        x[0], tfidf_(x[1], idf))).collectAsMap()
    del idf
    # del - Remove the variable if it won't be used again after using it and clean the memory space.

    def aggregate_business_words(line, bus_profile: dict):
        busid_list = line[1]
        user_profile = []
        for busid in busid_list:
            user_profile += bus_profile[busid]
        user_profile = set(user_profile)
        return (line[0], user_profile)
    # user_profile_dic = {'user1': [word2, word8, word24,....],.....}
    user_profile = textRDD.map(lambda x: (x['user_id'], {x['business_id']})).reduceByKey(
        lambda a, b: a.union(b)).map(lambda data: aggregate_business_words(data, business_profile))
    user_profile = user_profile.collectAsMap()

    time_end = time.time()
    print('Duration:', time_end - time_start)
    save_model(business_profile, user_profile, output_file_name)


if __name__ == "__main__":
    main()

    # for line in output_result:
    #     bus1 = line[0]
    #     bus2 = line[1]
    #     sim = line[2]
    #     out_line = {"b1": bus1, "b2": bus2, "sim": sim}
