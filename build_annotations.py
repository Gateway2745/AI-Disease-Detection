import os
import cv2

base_dir = '/home/rohit-ubuntu18_04/bboxes/new_dataset/'
hl_dir = os.path.join(base_dir, 'leaf_healthy')
dl_dir = os.path.join(base_dir, 'leaf_diseased')
he_dir = os.path.join(base_dir, 'ear_healthy')
de_dir = os.path.join(base_dir, 'ear_diseased')

labels_file = '/home/rohit-ubuntu18_04/bboxes/labels.csv'
log_file = '/home/rohit-ubuntu18_04/bboxes/log'
ann_dict_file = '/home/rohit-ubuntu18_04/annotations_dict.csv'

ann_dict = {}
with open(ann_dict_file, 'r') as f:
    for line in f.readlines():
        line = line.strip().split(',')
        ann_dict[line[0]] = line[1]
        
        
ann = open(labels_file, "r")
lines = ann.readlines()
ann_file = open('all_annotations.csv', 'w')

i=0
while(i<len(lines)):
    flag = 0
    img_data=lines[i].strip().split(',')
    img = cv2.imread(os.path.join(base_dir, '..', ann_dict[img_data[0]]))
    if(img.shape[0]>img.shape[1]):
        flag =1
        
    next_box = lines[i].strip().split(',')
    if(flag==1):
        [a,b,c,d] = next_box[2:]
        next_box[2] = int(img.shape[1])- int(d)
        next_box[3] = int(a)
        next_box[4] = int(img.shape[1])- int(b)
        next_box[5] = int(c)     

    file_name = os.path.basename(img_data[0])
    status=None
    if('lh' in file_name):
        status=0
    elif('ld' in file_name):
        status=1
    elif('eh' in file_name):
        status=2
    elif('ed' in file_name):
        status=3

    ann_file.write(f"{ann_dict[img_data[0]]},{status},{next_box[2]},{next_box[3]},{next_box[4]},{next_box[5]}\n")

    i+=1

ann_file.close()
ann.close()