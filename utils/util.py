import io
import csv
import os

dir = "../Datasets/Ms-celeb/"
saveFile = "MS-Celeb-1M_list_name.tsv"

with io.open(dir + saveFile, 'w', encoding='utf-8') as saveF:
    with io.open(dir + "Top1M_MidList.Name.tsv", 'r', encoding='utf-8') as tsvF:
        reader = csv.reader(tsvF, delimiter='\t')
        for row in reader:
            MID, name = row[0], row[1]
            if name.endswith("@en") and os.path.exists("D:/MS-Celeb-1M/" + MID):
                saveF.write(MID + '\t' + name[:-3] + '\n')