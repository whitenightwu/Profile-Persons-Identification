import csv
import io
import os
import shutil

dir = "D:\msceleb1m\MS-Celeb-1M\\"
washFile = "D:\msceleb1m\MS-Celeb-1M_list_name_without_cfp.tsv"

def wash_msceleb_remove_img_and_folders():
    file = []
    folders = []
    with io.open(washFile, 'r', encoding='utf-8') as f:
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

    with io.open(washFile, 'r', encoding='utf-8') as f:
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
            dest = dir + '\\' + dir1
            print(dest)
            shutil.rmtree(dest, ignore_errors=True)


def create_new_list_MSCeleb_without_cfp_and_exist_folders():
    saveFile = "D:\msceleb1m\MS-Celeb-1M_list_name_without_cfp_and_exist.tsv"
    with io.open(saveFile, 'w', encoding='utf-8') as saveF:
        with io.open(washFile, 'r', encoding='utf-8') as washF:
            reader = csv.reader(washF, delimiter='\t')
            for row in reader:
                MID, name = row[0], row[1]
                if os.path.exists(dir + MID):
                    saveF.write(row[0] + '\t' + row[1] + '\n')

def check_repeat():
    matchesFile = "matches_with_lfw.txt"
    saveFile = "checked_lfw.txt"
    hash = dict()

    with io.open(matchesFile, 'r', encoding='utf-8') as matchF:
        with io.open(saveFile, 'w', encoding='utf-8') as saveF:
            for line in matchF:
                MID = line.strip().split(' ')[0]
                if MID in hash:
                    continue
                else:
                    hash[MID] = 1
                    saveF.write(line)


def create_sample(count=5000):
    matchesFile = "matches_with_lfw.txt"
    saveFile = "sample.txt"
    mscelebFile = "D:\msceleb1m\MS-Celeb-1M_list_name_without_cfp_and_exist.tsv"

    hash = dict()
    with io.open(saveFile, 'w', encoding='utf-8') as saveF:
        with io.open(matchesFile, 'r', encoding='utf-8') as matchF:
            for line in matchF:
                count -= 1
                MID = line.strip().split(' ')[0]
                hash[MID] = 1
                saveF.write(line)

        with io.open(mscelebFile, 'r', encoding='utf-8') as msF:
            reader = csv.reader(msF, delimiter='\t')
            for row in reader:
                if count > 0:
                    MID = row[0]
                    if MID in hash:
                        continue
                    else:
                        count -= 1
                        hash[MID] = 1
                        saveF.write(row[0]+' ' + row[1] + '\n')
                else:
                    break





create_sample()