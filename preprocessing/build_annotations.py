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

extra_annotations = [
    ['../dataset/ears/healthy/hiren/20190907_134247.jpg',2,21,396,317,1409],
    ['../dataset/ears/healthy/hiren/20190907_134247.jpg',2,838,652,2304,4109],
    ['../dataset/ears/healthy/hiren/20190907_134247.jpg',2,664,1518,878,2161],
    ['../dataset/ears/healthy/hiren/20190907_134247.jpg',2,1143,1439,1257,1787],
    ['../dataset/ears/healthy/hiren/20190907_134247.jpg',2,734,518,930,965],
    ['../dataset/ears/healthy/hiren/20190907_134247.jpg',2,356,965,473,1331],
    ['../dataset/ears/healthy/hiren/20190907_134247.jpg',2,1160,618,1317,957],
    ['../dataset/ears/healthy/hiren/20190907_134247.jpg',2,1351,674,1512,1052],
    ['../dataset/ears/diseased/hiren/20190908_121320.jpg',3,208,431,2273,3222],
    ['../dataset/leaves/diseased/hiren/20190908_131607.jpg',1,393,1135,4127,2322],
    ['../dataset/ears/diseased/hiren/20190908_115804.jpg',3,25,722,1999,3013],
    ['../dataset/leaves/healthy/manan/20190908_102427.jpg',0,451,539,1682,4128],
    ['../dataset/leaves/healthy/manan/20190908_102427.jpg',0,599,283,3091,935],
    ['../dataset/ears/diseased/manan/20190908_110308.jpg',3,1617,1792,2151,3557],
    ['../dataset/ears/diseased/manan/20190908_110308.jpg',2,1495,909,1765,1605],
    ['../dataset/ears/diseased/manan/20190908_110308.jpg',2,2004,1018,2104,1435],
    ['../dataset/ears/diseased/manan/20190908_110308.jpg',2,2169,1144,2408,1522],
    ['../dataset/ears/diseased/manan/20190908_110308.jpg',2,2647,1996,2834,2800],
    ['../dataset/ears/diseased/manan/20190908_110308.jpg',2,221,1748,343,2265],
    ['../dataset/leaves/diseased/hiren/20190908_135425.jpg',1,578,26,1164,4128],
    ['../dataset/leaves/diseased/hiren/20190908_135425.jpg',1,47,2700,2208,3118],
    ['../dataset/ears/healthy/manan/20190907_133635.jpg',2,1,287,912,4128],
]
    
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

for ea in extra_annotations:
    ea = [str(x) for x in ea]
    line = ','.join(ea)
    ann_file.write(f'{line}\n')
    
ann_file.close()
ann.close()