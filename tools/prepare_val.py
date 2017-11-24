list_train='/home/dereyly/ImageDB/cdiscount/train2_prd.txt'
list_test='/home/dereyly/ImageDB/cdiscount/val2_prd.txt'

val='/home/dereyly/progs/pytorch_cdiscount/release/training_code/train_data/valid_id_v0_5000'

f_train=open('/home/dereyly/ImageDB/cdiscount/train_sync.txt','w')
f_test=open('/home/dereyly/ImageDB/cdiscount/val_sync.txt','w')
data_all={}
with open(list_train,'r') as f:
    for line in f.readlines():
        key=line.split('/')[-1].split('_')[0]
        data_all[key]=line
with open(list_test,'r') as f:
    for line in f.readlines():
        key=line.split('/')[-1].split('.')[0]
        data_all[key]=line
print('len',len(data_all))
count=0
with open(val,'r') as f:
    for line in f.readlines():
        count+=1
        try:
            key=line.strip()
            f_test.write(data_all[key])
            del data_all[key]
        except:
            pass
            # print(count)

for key, val in data_all.items():
    f_train.write(data_all[key])

print('len',len(data_all))