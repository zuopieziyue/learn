import math

doc_path = "D:\\mygithub\\data\\learn\\python\\hmm_model\\allfiles.txt"
mod_path = "D:\\mygithub\\data\\learn\\python\\hmm_model\\hmm_model.data"

#初始化模型参数
#其中s状态为：B M E S
STATUS_NUM = 4

def get_word_ch(word):
	ch_lst = []
	i = 0
	word_len = len(word)
	while i < word_len:
		ch_lst.append(word[i])
		i += 1
	return ch_lst

#1.初始概率
pi = [0.0 for col in range(STATUS_NUM)]
pi_sum = 0.0

#2.状态转移概率
A = [[0.0 for col in range(STATUS_NUM)] for row in range(STATUS_NUM)]
A_sum = [0.0 for col in range(STATUS_NUM)]

#3.发射概率
B = [dict() for col in range(STATUS_NUM)]
B_sum = [0.0 for col in range(STATUS_NUM)]

#打开文件，读入每一行
f_txt = open(doc_path, 'r', encoding='utf-8')

while True:
	line = f_txt.readline()
	if not line:
		break
	line = line.strip()
	if len(line) < 1 :
		continue
	words = line.split()
	
	ch_lst = []
	status_lst = []
	#获取一句话中每个字符对应的B、M、E、S[0, 1, 2, 3]状态
	for word in words:
		cur_ch_lst = get_word_ch(word=word)
		cur_ch_num = len(cur_ch_lst)
		#初始化字符状态0
		cur_status_lst = [0 for ch in range(cur_ch_num)]
		#只有单个字的状态为3：S
		if cur_ch_num == 1:
			cur_status_lst[0] = 3
		else:
			#标识0：B
			cur_status_lst[0] = 0
			#标识2：E
			cur_status_lst[-1] = 2
			#标识1：M
			for i in range(1,cur_ch_num-1):
				cur_status_lst[i] = 1
		#一行的所欲word放到ch_lst，状态status_lst
		ch_lst.extend(cur_ch_lst)
		status_lst.extend(cur_status_lst)

	for i in range(len(ch_lst)):
		cur_status = status_lst[i]
		cur_ch = ch_lst[i]
		#存储初始概率
		if i == 0:
			pi[cur_status] += 1.0
			pi_sum += 1.0
		#存储发射概率B
		if cur_ch in B[cur_status]:
			B[cur_status][cur_ch] += 1.0
		else :
			B[cur_status][cur_ch] = 1.0
		B_sum[cur_status] += 1.0
		#存储转移概率A
		if i+1 < len(ch_lst)-1:
			A[cur_status][status_lst[i+1]] += 1.0
			A_sum[cur_status] += 1.0

f_txt.close()

#将统计结果转化为概率形式
for i in range(STATUS_NUM):
	pi[i] /= pi_sum
	#A
	for j in range(STATUS_NUM):
		A[i][j] /= A_sum[i]
	for ch in B[i]:
		B[i][ch] /= B_sum[i]

#存储模型->模型文件，将概率转化成log形式
f_mod = open(mod_path, "w", encoding='utf-8')
for i in range(STATUS_NUM):
	if pi[i] != 0.0:
		log_p = math.log(pi[i])
	else:
		log_p = 0.0
	f_mod.write(str(log_p) + " ")
f_mod.write("\n")

#A转移矩阵
for i in range(STATUS_NUM):
	for j in range(STATUS_NUM):
		if A[i][j] != 0.0:
			log_p = math.log(A[i][j])
		else:
			log_p = 0.0
		f_mod.write(str(log_p) + " ")
f_mod.write("\n")

#B发射概率
for i in range(STATUS_NUM):
	for ch in B[i]:
		if B[i][ch] != 0.0:
			log_p = math.log(B[i][ch])
		else:
			log_p = 0.0
		f_mod.write(ch + " " + str(log_p) + " ")
f_mod.write("\n")



