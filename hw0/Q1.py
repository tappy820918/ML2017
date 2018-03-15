import sys
def main(*args):
	path = args[0][1]
	words_key = list()
	with open(path, 'rb') as f:
		words_list = f.read().decode().split()
	for word in words_list:
		if word not in words_key:
			words_key.append(word)
	number = [0]*len(words_key)
	with open('./Q1.txt', 'w') as f:
		for i in range(300):
			f.write(words_key[i])
			f.write('\t')
			f.write(str(i))
			f.write('\t')
			f.write(str(words_list.count(words_key[i])))
			f.write('\n')

if __name__ == '__main__':
	main(sys.argv)