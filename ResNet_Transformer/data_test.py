import pickle
import os

train = []
val = []

with open("grid.pickle", 'rb') as file:
    data = pickle.load(file)

with open('data_test.txt', 'w') as file:
    file.write(str(data))
    samples = data['train']
    for i in range(len(samples)):
        sample = samples[i]
        dir = sample['images'].split('/')
        out_dir = os.path.join(dir[1], dir[7], dir[6], dir[9])
        # i = out_dir
        print(out_dir)
        sample['images'] = str(out_dir)
        train.append(out_dir)

with open('testing.txt', 'w') as file:
    for i in train:
        file.write(i)
        file.write('\n')

with open("grid.pickle", 'rb') as file:
    data = pickle.load(file)
    samples = data['train']
    for i in samples:
        dir = i['images'].split('/')
        out_dir = os.path.join(dir[1], dir[7], dir[6], dir[9])
        print(out_dir)
        train.append(out_dir)

with open("grid.pickle", 'rb') as file:
    data = pickle.load(file)
    samples = data['val']
    for i in samples:
        dir = i['images'].split('/')
        out_dir = os.path.join(dir[1], dir[7], dir[6], dir[9])
        print(out_dir)
        val.append(out_dir)

with open("grid.pickle", 'wb') as file:
    file.dump()

with open("data_test.txt", 'w') as file:
    file.write(str(data))
    # for i in train:
    #     file.write(i)
    #     file.write('\n')

    