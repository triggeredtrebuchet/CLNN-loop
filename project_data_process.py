import json
import os
import numpy as np

data = json.load(open("data/project_data/dev.json"))
right_x = [data[i][0] for i in data]
left_x = [data[i][1] for i in data]
y = [data[i][2] for i in data]

# write right_x as a .fasta file
with open("data/project_data/dev_right.fasta", "w") as f:
    for i in right_x:
        f.write(">+\n")
        f.write(i)
        f.write("\n")

# write left_x as a .fasta file
with open("data/project_data/dev_left.fasta", "w") as f:
    for i in left_x:
        f.write(">-\n")
        f.write(i)
        f.write("\n")

np.save('data/project_data/dev_y.npy', np.array(y))