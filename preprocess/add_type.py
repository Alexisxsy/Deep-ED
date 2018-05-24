import codecs
from collections import defaultdict, Counter

def get_razor(fname):
	max_type_num = 0
	with codecs.open(fname, "r", encoding = "utf-8", errors = 'ignore') as f:
		name_type_map = defaultdict(set)
		for line in f:
			tokens = line.strip().split("\t")
			name_type_map[tokens[0]] = tokens[1:]
			max_type_num = max(max_type_num, len(tokens[1:]))
	print("[INFO] number of entities:{}".format(len(name_type_map.keys())))
	print("[INFO] max type num :{}".format(max_type_num))
	return name_type_map

def get_type2id(type_file):
	with open(type_file, "r") as f:
		type_id_map = {}
		for line in f:
			idx, typ, _ = line.strip().split("\t")
			type_id_map[typ] = int(idx)
	print("[INFO] number of types:{}".format(len(type_id_map.keys())))
	return type_id_map, len(type_id_map.keys())

def get_all_type_vec(fname):
	with open(fname, "r", encoding="utf-8") as f:
		all_type_vec = [line.replace("\n","") for line in f]
	return all_type_vec

def get_type_vec(type_list):
	vec = ['0' for i in range(type_num)]
	if type_list[0] != "NoneType":
		for typ in type_list:
			vec[type_id_map[typ]] = '1'
	return ",".join(vec)

def get_type_code(type_list):
	type_code = ["999" for i in range(8)]
	if type_list[0] != 'NoneType':
		assert(len(type_list) <= 8)
		for i in range(len(type_list)):
			type_code[i] = str(type_id_map[type_list[i]])
			type_code[i] = '0' * (3-len(type_code[i])) + type_code[i] # gurantee there are 3 digits each 
	# print(type_code)
	assert(len("".join(type_code)) == 24)
	return "".join(type_code)

def add_type_train(fname, fsave):
	fin = open(fname, "r")
	fout = open(fsave, "w+")
	tot_ent = set()
	trigger_ent = set()
	for line in fin:
		tokens = line.strip().split("\t")
		assert tokens[5] == "CANDIDATES"
		assert tokens[-2] == "GT:"
		gold_info = tokens[-1].split(",")
		gold_type = name_type_map.get(gold_info[-1], ["NoneType"])
		gold_vec = get_type_vec(gold_type)
		gold_type_code = get_type_code(gold_type)
		gold_str = ",".join([gold_info[0], gold_info[1], gold_type_code, gold_info[-1]])
		all_type = []
		for candid_info in tokens[6:-2]:
			# print(candid_info)
			candid_info = candid_info.split(",")
			wikiid = candid_info[0]
			p = candid_info[1]
			name = ",".join(candid_info[2:])
			cur_type = name_type_map.get(name, ["NoneType"])
			type_code = get_type_code(cur_type)
			tot_ent.add(name)
			if cur_type != ["NoneType"]:
				trigger_ent.add(name)
			# print([name] + cur_type)
			all_type.append(",".join([wikiid, type_code, name]))
		rs = [tokens[0]] + all_type + ["GT:", gold_str, gold_vec]
		fout.write("\t".join(rs) + "\n")

	fin.close()
	fout.close()
	print("[INFO] number of entities:{}, number of entities that have type:{}".format(len(tot_ent), len(trigger_ent)))

def add_type_test(fname, fsave, ftype_vec):#ctx_vec contains all context vector, each line contains n types with their corresponding score
	type_vec = get_all_type_vec(ftype_vec)
	fin = open(fname, "r")
	fout = open(fsave, "w+")
	tot_ent = set()
	trigger_ent = set()
	line_num = 0
	for line in fin:
		tokens = line.strip().split("\t")
		# print("[DEBUG]" + tokens[5])
		assert tokens[5] == "CANDIDATES"
		assert tokens[-2] == "GT:"
		gold_info = tokens[-1].split(",")
		if len(gold_info) == 1:
			gold_str = "-1"
		else:
			gold_type = name_type_map.get(gold_info[-1], ["NoneType"])
			gold_type_code = get_type_code(gold_type)
			gold_str = ",".join([gold_info[0], gold_info[1], gold_type_code, gold_info[-1]])
		#gold vec comes from prediction confident scores
		gold_vec = type_vec[line_num]
		all_type = []
		for candid_info in tokens[6:-2]:
			candid_info = candid_info.split(",")
			if candid_info[0] == "EMPTYCAND":
				rs = [tokens[0]] + ["EMPTYCAND"] + [gold_vec]
				break
			else: 
				wikiid = candid_info[0]
				p = candid_info[1]
				name = ",".join(candid_info[2:])
				cur_type = name_type_map.get(name, ["NoneType"])
				type_code = get_type_code(cur_type)
				tot_ent.add(name)
				if cur_type != ["NoneType"]: 
					trigger_ent.add(name)
				all_type.append(",".join([wikiid, type_code, name]))
		rs = [tokens[0]] + all_type + ["GT:", gold_str, gold_vec]
		fout.write("\t".join(rs) + "\n")

		line_num += 1

	fin.close()
	fout.close()
	print("[INFO] number of entities:{}, number of entities that have type:{}".format(len(tot_ent), len(trigger_ent)))

if __name__ == "__main__":
	type_id_map, type_num = get_type2id("../../../NFGEC/resource/conll/label2id_conll.txt")
	name_type_map =  get_razor("../../../NFGEC/data/type/razor_figer.tsv")
	add_type_train("../../data_path/generated/test_train_data/aida_train.csv", "../../data_path/generated/test_train_data/aida_train_type.csv")
	
	add_type_test("../../data_path/generated/test_train_data/aida_testA.csv", 
		"../../data_path/generated/test_train_data/aida_testA_type.csv",
		"../../../NFGEC/result/conll_lstm_False_False_score_ta.tsv")
	
	add_type_test("../../data_path/generated/test_train_data/aida_testB.csv", 
		"../../data_path/generated/test_train_data/aida_testB_type.csv",
		"../../../NFGEC/result/conll_lstm_False_False_score_tb.tsv")