ad = open('annotations_dict.csv', 'w')
with open(log_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        pro = line.strip().split('->')
        if(len(pro)>1 and len(pro[1])>5):   # avoid images with no mappings
            pro[0] = pro[0].replace('bajra_dataset_copy', 'dataset')
            print(pro[1])
            ad.write('{},{}\n'.format(pro[1].strip(), pro[0].strip()))
    
ad.close()
