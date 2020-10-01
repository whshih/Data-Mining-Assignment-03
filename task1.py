from pyspark import SparkContext
import json
from itertools import combinations
from random import sample
import time
import sys

def tojson(line):
    lines = line.splitlines()
    data = json.loads(lines[0])
    return data

def Convert(tup):
    tuple_list = []
    dic = {}
# List to Dictionary {key: (string) business, value:(list) user}
    for b, u in tup:  # [('business', ['user1','user2',...]),(...)]
        dic[b] = u  # {{"business 1":['user1','user2',...]},{...}}

# Dictionary to Tuple List = [('business','user'),(...),...] --> len(tuple_list) = 506685
    for b, u_list in dic.items():
        for u in u_list:
            tuple_list.append((b, u))
    return tuple_list


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


def minhash(user_id_vector: list, rand_num_a: list, user_dic: dict, user_list: list):
    # Dictionary with row number (total): user_dic = {'---1lKK3aKOuomHnwAkAow': 0, '--2vR0DIsmQ6WfcSzKWigw': 1,....}
    # List of user_id containing value of 1: user_id list per business = [user_id,...]
    origin_row_num = []
    for element in user_id_vector:
        origin_row_num.append(user_dic[element])

    b = 1356  # number of bends
    prime = 743111
    total_users = len(user_list)
    result = []
    for a in rand_num_a:
        new_row_num = []
        for x in origin_row_num:
            # Generate new row num for each user whose value is 1.
            hashed_s = ((a*x + b) % prime) % total_users
            new_row_num.append(hashed_s)
        result.append(min(new_row_num))
    return result

def rand_num_generator(sign_num):
    para_for_hash_func = sample(range(1, 5*sign_num), sign_num)
    return para_for_hash_func

def band(signature: list, band: int, r: int):
    #signature = [(business, [sig,sig,...]),...]
    band_v_business = []  # [((band_num, [sig...]), business)]
    for i in range(band):
        band_v_business.append(
            ((i, tuple(signature[1][i*r:(i+1)*r])), [signature[0]]))
    return band_v_business

def jaccard(similar_pair: list, bus_u: dict):
    #similar_pair = [(bus_sim_A, bus_sim_B), (),...]
    # bus_u = {'business': [user567, user123,...],...}
    b1 = set(bus_u[similar_pair[0]])  # set([user, user, ...]) for bus_sim_A
    b2 = set(bus_u[similar_pair[1]])  # set([user, user, ...]) for bus_sim_B
    jac = len(b1 & b2) / len(b1 | b2)
    return (similar_pair[0], similar_pair[1], jac)

def main():
    time_start = time.time()
    sign_num = 1000  # number of hash func
    sc = SparkContext('local[*]', 'task1')
    data = 'train_review.json' #sys.argv[1]
    output_file_name = 't1_result.json' #sys.argv[2]
    textRDD = sc.textFile(data).map(tojson)
    # Total number of user_id = 26184 (user_list -> user_dic)

    user_list = sorted(textRDD.map(lambda key: (
        key['user_id'], 1)).distinct().collect())
    user_dic = list2dic(user_list)
    # Total number of business_id = 10253 (bus_list -> bus_dic)

    bus_list = sorted(textRDD.map(lambda key: (
        key['business_id'], 1)).distinct().collect())
    bus_dic = list2dic(bus_list)

    rand_num_list = rand_num_generator(sign_num)
    # The number of (business, user) = 488560 --> [(business1, user1),(business2, user2),...]
    pair = textRDD.map(lambda key: (key['business_id'], [key['user_id']])).reduceByKey(
        lambda y, z: y + z).sortByKey()
    b_u = pair.collect()
    bus_user_dic = dict(b_u)

    signature = pair.map(lambda a: (a[0], minhash(
        a[1], rand_num_list, user_dic, user_list)))

    b = 500
    r = int(sign_num/b)  # num of rows = 3
    # Find similar pair(len > 2) and compare every two band if it is consist then assign to same bucket
    # 15000~20000
    candidate = signature.flatMap(lambda x: band(x, b, r)).reduceByKey(lambda a, b: a + b).filter(
        lambda candidate_pair: len(candidate_pair[1]) > 1).flatMap(
        lambda x: set(combinations(x[1], 2))).distinct()

    jaccard_result = candidate.map(lambda sim: jaccard(sim, bus_user_dic)).filter(lambda jac: jac[2] >= 0.05).sortBy(
        lambda business: business[1]).sortBy(lambda business: business[0])
    output_result = jaccard_result.collect()

    output_file = open(output_file_name, 'w')
    for line in output_result:
        bus1 = line[0]
        bus2 = line[1]
        sim = line[2]
        out_line = {"b1": bus1, "b2": bus2, "sim": sim}
        json.dump(out_line, output_file)
        output_file.write('\n')

    output_file.close()
    time_end = time.time()
    print('Duration:', time_end - time_start)


if __name__ == "__main__":
    main()
