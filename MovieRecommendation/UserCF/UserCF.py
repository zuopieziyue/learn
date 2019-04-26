# coding = utf-8

#基于用户的协同过滤推荐算法实现
import random

import path
from operator import itemgetter

class UserBasedCF():
    #初始化相关参数
    def __init__(self):
        #找到与目标用户兴趣相似的20个用户，为其推荐按10部电影
        self.n_sim_user = 20
        self.n_rec_moive = 10

        #将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        #用户相似度矩阵
        self.user_sim_matrix = {}
        self.moive_count = 0

        print('Similar user number = %d' % self.n_sim_user)
        print('Recommneded movie number = %d' % self.n_rec_movie)

    #读取文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            if i == 0:
                continue
            yield line.strip('\r\n')
        print('Load %s success!' % filename)

    #读取文件得到"用户-电影"数据
    def get_dataset(self, filename, pivot=0.75):
        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file(filename):
            user, moive, rating, timestamp = line.split(',')
            if random.random() < pivot:
                self.trainSet.setdefault(user, {})
                self.testSet[user][moive] = rating
                trainSet_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][moive] = rating
                testSet_len += 1
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)

    #计算用户之间的相似度
    def calc_user_sim(self):
        #构建"电影-用户"倒排索引
        #key = moiveID, value = list of userIDs who have seen this moive
        print('Building movie-user table ...')
        moive_user = {}
        for user, moives in self.trainSet.items():
            for moive in moives:
                if moive not in moive_user:
                    moive_user[moive] = set()
                moive_user[moive].add(user)
        print('Build movie-user table success!')

        self.moive_count = len(moive_user)
        print('Total movie number = %d' % self.movie_count)

        print('Build user co-rated movies matrix ...')
        for moive, users in moive_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1
        print('Build user co-rated movies matrix success!')

        #计算相似性
        print('Calculating user similarity matrix ...')
        for u, related_users in self.user_sim_matrix.items():
            for v, count in related_users.items():
                self.user_sim_matrix[u][v] = count / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))
        print('Calculate user similarity matrix success!')

    #针对目标用户U，找到其最相似的K个用户，产生N个推荐
    def recommend(self, user):
        K = self.n_sim_user
        N = self.n_rec_moive
        rank = {}
        watched_moives = self.trainSet[user]

        # v = similar user, wuv=similar factor
        for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]
            for moive in self.trainSet[v]:
                if moive in watched_moives:
                    continue
                rank.setdefault(moive, 0)
                rank[moive] += wuv
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    #产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print("Evaluation start ...")
        N = self.n_rec_moive
        #准确率和召回率
        hit = 0
        rec_count = 0
        rest_count = 0
        #覆盖率
        all_rec_movies = set()

        for i, user in enumerate(self.trainSet):
            test_moives = self.testSet.get(user, {})
            rec_moives = self.recommend(user)
            for moive, w in rec_moives:
                if moive in test_moives:
                    hit += 1
                all_rec_moives.add(moive)
            rec_count += N
            test_count += len(test_moives)
        
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))

if __name__ == '__main__':
    rating_file = 'D:\\学习资料\\推荐系统\\ml-latest-small\\ratings.csv'
    userCF - UserBasedCF()
    userCF.get_dataset(rating_file)
    userCF.calc_user_sim()
    userCF.evaluate()
