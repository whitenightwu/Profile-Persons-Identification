import io
import os
import shutil

dir = "D:\msceleb1m\MS-Celeb-1M\\"
washF = "D:\msceleb1m\MsCeleb1M_washlist.txt"
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


#method_name()

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

print("HI!!!")

for dir1 in os.listdir(dir):
    if dir1 not in folders:
        dest = dir+ '\\' + dir1
        print(dest)
        shutil.rmtree(dest, ignore_errors=True)