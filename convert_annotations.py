bt = open('raw_to_pro.csv', 'w')
with open(log_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        pro = line.strip().split('->')
        if(len(pro)>1):
            pro[0] = pro[0].replace('bajra_dataset_copy', 'dataset')
            print(pro[0])
            bt.write('{},{}\n'.format(pro[1].strip(), pro[0].strip()))
    
bt.close()