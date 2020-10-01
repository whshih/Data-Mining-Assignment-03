import json
import time
from math import sqrt
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

def cosine_sim(pair, business_profile: dict, user_profile: dict):
    # business_profile_dic = {'business1': set('located','next',...),....}
    # user_profile_dic = {'user1': set(word2, word8, word24,....),.....}
    #print(1)
    user_id = pair[0]
    business_id = pair[1]
    try:
        a = business_profile[business_id]  # set([user, user, ...]) for bus_sim_A
        b = user_profile[user_id]  # set([user, user, ...]) for bus_sim_B
    except KeyError:
        return (user_id, business_id, 0)
    cos_sim = len(b.intersection(a)) / sqrt(len(a)*len(b))
    return (user_id, business_id, cos_sim)

# def cosine_sim(user_bus, business_profile: dict, user_profile: dict):
#     # business_profile_dic = {'business1': set('located','next',...),....}
#     # user_profile_dic = {'user1': set(word2, word8, word24,....),.....}
#     a = business_profile[user_bus[0]]
#     b = user_profile[user_bus[1]]
#     cos_sim = len(a & b) / (sqrt(len(a))*sqrt(len(b)))
#     return (user_bus[0], user_bus[1], cos_sim)

def main():
    time_start = time.time()
    sc = scc()

    data = sys.argv[1] #'test_review.json'
    model_path = sys.argv[2] #'task2.model'
    output_file_name = sys.argv[3] #'t2_result.json'

    testRdd = sc.textFile(data).map(tojson)

    def load_model(file_path):
        import pickle
        with open(file_path, 'rb') as m:
            business_profile = pickle.load(m)
            user_profile = pickle.load(m)
        return business_profile, user_profile

    business_profile, user_profile = load_model(model_path)

    sim = testRdd.map(lambda x: (x['user_id'], x['business_id'])).map(
        lambda x: cosine_sim(x, business_profile, user_profile)).filter(
        lambda x: 1 if x[2] >= 0.01 else 0).collect()
    output_file = open(output_file_name, 'w')
    for line in sim:
        user = line[0]
        bus = line[1]
        cos_sim = line[2]
        out_line = {"user_id": user, "business_id": bus, "sim": cos_sim}
        json.dump(out_line, output_file)  # (out_line, output_file)
        output_file.write('\n')

    output_file.close()
    time_end = time.time()
    print('Duration:', time_end - time_start)

if __name__ == "__main__":
    main()
