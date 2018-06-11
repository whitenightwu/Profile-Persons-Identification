import io
import os
import shutil
import csv

dir = "D:\msceleb1m\MS-Celeb-1M\\"
washFile = "D:\msceleb1m\MS-Celeb-1M_list_name_without_cfp.tsv"
saveFile = "D:\msceleb1m\MS-Celeb-1M_list_name_without_cfp_and_exist.tsv"
file = []
folders = []


def method_name():
    with io.open(washF, 'r', encoding='utf-8') as f:
        line = f.readline()
        prev_folder = line.strip().split('/')[0]
        file.append(line.strip().split('/')[1])
        i = 0

        for line in f.readlines():
            i += 1
            if i % 100 == 0:
                print(i)
            if line != "==":
                folder = line.strip().split('/')[0]
                if folder == prev_folder:
                    file.append(line.strip().split('/')[1])
                    continue

            x_dir = dir + prev_folder
            if os.path.exists(x_dir):
                folders.append(prev_folder)
                prev_folder = folder
                for f1 in os.listdir(x_dir):
                    if f1 not in file:
                        os.remove(x_dir + '\\' + f1)
                file.clear()
                if (line != "=="):
                    file.append(line.strip().split('/')[1])


def method_name2():
    with io.open(washF, 'r', encoding='utf-8') as f:
        line = f.readline()
        prev_folder = line.strip().split('/')[0]
        i = 0

        for line in f.readlines():
            i += 1
            if i % 1000 == 0:
                print(i)
            if line != "==":
                folder = line.strip().split('/')[0]
                if folder == prev_folder:
                    continue

            x_dir = dir + prev_folder
            if os.path.exists(x_dir):
                folders.append(prev_folder)
                prev_folder = folder
                file.clear()
    for dir1 in os.listdir(dir):
        if dir1 not in folders:
            dest = dir+ '\\' + dir1
            print(dest)
            shutil.rmtree(dest, ignore_errors=True)

with io.open(saveFile, 'w', encoding='utf-8') as saveF:
    with io.open(washFile, 'r', encoding='utf-8') as washF:
        reader = csv.reader(washF, delimiter='\t')
        for row in reader:
            MID, name = row[0], row[1]
            if os.path.exists(dir + MID):
                saveF.write(row[0] + '\t' + row[1] + '\n')