doc_path = ""

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
A = [0.0 for col in range(STATUS_NUM) for row in range(STATUS_NUM)]
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
	
	
	
f_txt.close()


