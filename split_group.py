import numpy as np
dataname = 'Gowalla'
datapath = 'data/' + dataname + '/train.txt'
inumofuser = []
with open(datapath, 'r') as f:
    for line in f.readlines():
        if line:
            values = line.strip().split(' ')
            user = int(values[0])
            items = values[1:]
            inumofuser.append([user, len(items)])

if dataname in ['AMusic', 'ABeauty']:
    g = [0, 4, 8, 16, 32]
else:
    g = [0, 8, 16, 32, 64]
inumofuser = np.array(inumofuser)
uidofgroup = {}
for i in range(len(g) - 1):
    index = np.where((inumofuser[:, 1] > g[i]) & (inumofuser[:, 1] <= g[i+1]))
    uidofgroup[i+1] = list(index[0])

index = np.where(inumofuser[:, 1] > g[-1])
uidofgroup[len(g)] = list(index[0])

for i in range(len(g)):
    print(f'{i+1}-group: {len(uidofgroup[i+1])}')
print('all user number', len(inumofuser))
# print(uidofgroup)
grouppath= 'data/' + dataname + '/group.txt'
f = open(grouppath, 'w')
f.write(str(uidofgroup))
f.flush()
f.close()