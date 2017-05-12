import random
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mpl_toolkits.mplot3d import Axes3D
ys = np.linspace(0,0,12)
xs = np.chararray((12))
def csv_reader():
    with open('list_annotated_img.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        rows = list(spamreader)
        totalrows = len(rows)
        print totalrows
        index = 0
        for row in spamreader:
            ys[index] = row[0]
            #xs[index] = row[1]
            index = index+1

mpl.rcParams['font.size'] = 10

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
csv_reader()

for z in [1,2,3,4,5,6,7,8,9,10,11,12]:
     xs = xrange(1,13)
     color =plt.cm.Set2(random.choice(xrange(plt.cm.Set2.N)))
     ys = ys*0.8
     ax.bar(xs, ys, zs=z, zdir='y', color=color, alpha=0.8)

ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xs))
ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ys))

ax.set_xlabel('Pollution')
ax.set_ylabel('Regions')
ax.set_zlabel('Index')

#plt.show()
