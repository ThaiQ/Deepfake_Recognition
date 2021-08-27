from thispersondoesnotexist import get_online_person, get_checksum_from_picture, save_picture
import os
import csv

#variable
image_counts=20000 #saving 20k images

#newfolder
newpath = r'./DoesnotExistData' 
csv_path = "./DoesnotExistData/DoesnotExistData_filename.csv"
if not os.path.exists(newpath):
    os.makedirs(newpath)
    os.makedirs(newpath+'./fake')
    os.makedirs(newpath+'./real')

saved = []
f = open(csv_path, "a")
with open(csv_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            saved.append(row[0])

i = 0
while (i < image_counts):
    # Using function
    picture = get_online_person()
    checksum2 = get_checksum_from_picture(picture)  # Method is optional, defaults to "md5"

    if checksum2 not in saved:
        saved.append(checksum2)
        f.write(checksum2+'\n')
        save_picture(picture, './DoesnotExistData/fake/{}.jpg'.format(checksum2))
        i += 1
    
    print('{} / {}'.format(i,image_counts))

f.close()