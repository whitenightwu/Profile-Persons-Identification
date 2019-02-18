import os
# path = r'C:\Users\zhizh\Desktop\face_near_infrared_test _v3'

# #########################
# path = '/home/ydwu/datasets/face_near_infrared_test _v5/INF'
# path_RGB = '/home/ydwu/datasets/face_near_infrared_test _v5/RGB'

# count = 1
# for person in os.listdir(path):
#     for each_img in os.listdir(path+'/'+person):
#         os.rename(path+'/'+person+'/'+each_img,path+'/'+person+'/'+str(count)+'.png')
#         os.rename(path_RGB+'/'+person+'/'+each_img,path_RGB+'/'+person+'/'+str(count)+'.png')
#         count+=1
#     count = 1




#########################
# path = '/home/ydwu/datasets/white-lfw'
path = '/home/ydwu/datasets/white-cfp-dataset-merge'

l=[]
for person in os.listdir(path):
    for each_img in os.listdir(path+'/'+person):
        # print(each_img)
        # xxx=person+'/'+each_img[:-4]
        # print(xxx)
        # l.append(person+'/'+each_img.strip('.jpg'))
        l.append(person+'/'+each_img[:-4])

yyy_1=0
yyy_2=0
count_1=1
count_2=1
with open('cfp_pair_part.txt','a') as f:
    for i in l:
        for j in l:
            if i != j and i.split('/')[0] == j.split('/')[0]:
                if count_1%4==1:#1:#count_1%10==1:
                    f.write(i.split('/')[0]+' '+i.split('/')[1]+' '+j.split('/')[1]+'\n')
                    yyy_1=yyy_1+1
                count_1=count_1+1
                
            elif i != j and i.split('/')[0] != j.split('/')[0]:
                if count_2%850==1:
                    f.write(i.split('/')[0]+' '+i.split('/')[1]+' '+j.split('/')[0]+' '+j.split('/')[1]+'\n')
                    yyy_2=yyy_2+1
                count_2=count_2+1

print(count_1)
print(count_2)
print(yyy_1)
print(yyy_2)
