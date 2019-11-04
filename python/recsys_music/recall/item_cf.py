import math
import operator

'''
input: train:{user_id:{item_id:rating(score)},...}
'''
def item_similarity(train):
    # 计算item1与item2相同的user的数量
    C = dict()  # 存item与item相同user的个数 分子
    N = dict()  # item的用户数量 分母
    for u, items in train.items():
        for i in items:
            if N.get(i, -1) == -1:
                N[i] = 0
            N[i] += 1
            if C.get(i, -1) == -1:
                C[i] = dict()
            for j in items:
                if i == j:
                    continue
                elif C[i].get(j, -1) == -1:
                    C[i][j] = 0
                C[i][j] += 1
    # 加分母计算相似度
    W = dict()
    for i, related_items in C.items():
        if W.get(i, -1) == -1:
            W[i] = dict()
        for j, cij in related_items.items():
            if W[i].get(j, -1) == -1:
                W[i][j] = 0
            W[i][j] += cij / math.sqrt(N[i] * N[j])
    return W


def recommend(train, user, w, k):
    rank = dict()
    ru = train[user]
    for i, pi in ru.items():
        for j, wj in sorted(w[i].items(),
                            key=operator.itemgetter(1),
                            reverse=True)[0:k]:
            if j in ru:
                continue
            elif rank.get(j, -1) == -1:
                rank[j] = 0
            rank[j] += pi * wj
    return rank
