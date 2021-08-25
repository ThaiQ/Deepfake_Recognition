import csv

with open('converted_path.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        line = row[0].split(',')
        print(line)

        img_dir = line[0]