import csv
import numpy as np

value = []
with open('data/genes_{list_id}.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)
    for line in reader:
        if np.shape(line)[0] > 1: 
            value.append(line[2]) 
    value = np.array(value)

with open('values.csv', 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    random = np.random.rand(np.shape(value.T)[0],)
    row = np.column_stack((value.T, random))
    writer.writerow(["Gene", "Value"])
    for entry in row:
        writer.writerow(entry)