
# coding: utf-8

# #DATASCI W261: Machine Learning at Scale

# # Data Generation

# 1.Generate data: 2.Three clusters 3.True centroids (4,0), (6,6), (0,4)

# In[6]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import pylab 
size1 = size2 = size3 = 10000
samples1 = np.random.multivariate_normal([4, 0], [[1, 0],[0, 1]], size1)
data = samples1
samples2 = np.random.multivariate_normal([6, 6], [[1, 0],[0, 1]], size2)
data = np.append(data,samples2, axis=0)
samples3 = np.random.multivariate_normal([0, 4], [[1, 0],[0, 1]], size3)
data = np.append(data,samples3, axis=0)
# Randomlize data
data = data[np.random.permutation(size1+size2+size3),]
np.savetxt('Kmeandata.csv',data,delimiter = ",")


# # Data Visualiazation

# In[7]:

pylab.plot(samples1[:, 0], samples1[:, 1],'*', color = 'red')
pylab.plot(samples2[:, 0], samples2[:, 1],'o',color = 'blue')
pylab.plot(samples3[:, 0], samples3[:, 1],'+',color = 'green')
pylab.show()


# # MrJob class for Kmeans

# ### If you want to change the code, please edit Kmeans.py directly

# In[8]:

get_ipython().run_cell_magic(u'writefile', u'Kmeans.py', u'from numpy import argmin, array, random\nfrom mrjob.job import MRJob\nfrom mrjob.step import MRStep\nfrom itertools import chain\nimport os\n\n#Calculate find the nearest centroid for data point \ndef MinDist(datapoint, centroid_points):\n    datapoint = array(datapoint)\n    centroid_points = array(centroid_points)\n    diff = datapoint - centroid_points \n    diffsq = diff*diff\n    # Get the nearest centroid for each instance\n    minidx = argmin(list(diffsq.sum(axis = 1)))\n    return minidx\n\n#Check whether centroids converge\ndef stop_criterion(centroid_points_old, centroid_points_new,T):\n    oldvalue = list(chain(*centroid_points_old))\n    newvalue = list(chain(*centroid_points_new))\n    Diff = [abs(x-y) for x, y in zip(oldvalue, newvalue)]\n    Flag = True\n    for i in Diff:\n        if(i>T):\n            Flag = False\n            break\n    return Flag\n\nclass MRKmeans(MRJob): # iteration step\n    self.centroid_points=[] # not necessary unless self\n    self.k=3    \n    def steps(self):\n        return [\n            MRStep(mapper_init = self.mapper_init, mapper=self.mapper,combiner = self.combiner,reducer=self.reducer)\n               ]\n    #load centroids info from file\n    def mapper_init(self):\n        print "Current path:", os.path.dirname(os.path.realpath(__file__))\n        \n        self.centroid_points = [map(float,s.split(\'\\n\')[0].split(\',\')) for s in open("Centroids.txt").readlines()]\n        #open(\'Centroids.txt\', \'w\').close()\n        \n        print "Centroids: ", self.centroid_points\n        \n    #load data and output the nearest centroid index and data point \n    def mapper(self, _, line):\n        D = (map(float,line.split(\',\'))) # list of numbers\n        yield int(MinDist(D,self.centroid_points)), (D[0],D[1],1) # 1 at end is to sum them up in the reducer\n    #Combine sum of data points locally\n    def combiner(self, idx, inputdata):\n        sumx = sumy = num = 0\n        for x,y,n in inputdata:\n            num = num + n\n            sumx = sumx + x\n            sumy = sumy + y\n        yield idx,(sumx,sumy,num)\n    #Aggregate sum for each cluster and then calculate the new centroids\n    def reducer(self, idx, inputdata): \n        centroids = []\n        num = [0]*self.k \n        for i in range(self.k):\n            centroids.append([0,0])\n        for x, y, n in inputdata:\n            num[idx] = num[idx] + n\n            centroids[idx][0] = centroids[idx][0] + x\n            centroids[idx][1] = centroids[idx][1] + y\n        centroids[idx][0] = centroids[idx][0]/num[idx]\n        centroids[idx][1] = centroids[idx][1]/num[idx]\n\n        yield idx,(centroids[idx][0],centroids[idx][1])\n      \nif __name__ == \'__main__\':\n    MRKmeans.run()')


# # Driver:

# Generate random initial centroids
# 
# New Centroids = initial centroids
# 
# While(1)：
# + Cacluate new centroids
# + stop if new centroids close to old centroids
# + Updates centroids 

# In[11]:

get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')
from numpy import random
from Kmeans import MRKmeans, stop_criterion
mr_job = MRKmeans(args=['Kmeandata.csv', '--file=Centroids.txt']) # training data, initial centriods coded below

#Geneate initial centroids
centroid_points = []
k = 3
for i in range(k):
    centroid_points.append([random.uniform(-3,3),random.uniform(-3,3)])
with open('Centroids.txt', 'w') as f:
        f.writelines(','.join(str(j) for j in i) + '\n' for i in centroid_points)

# Update centroids iteratively
i = 0
while(1):
    # save previous centoids to check convergency
    centroid_points_old = centroid_points[:] # store the current version of the centroids
    print "iteration"+str(i)+":"
    with mr_job.make_runner() as runner: 
        runner.run()
        # stream_output: get access of the output 
        for line in runner.stream_output():
            key,value =  mr_job.parse_output_line(line)
            print key, value
            centroid_points[key] = value
            
        # Update the centroids for the next iteration
        with open('Centroids.txt', 'w') as f:
            f.writelines(','.join(str(j) for j in i) + '\n' for i in centroid_points)
        
    print "\n"
    i = i + 1
    if(stop_criterion(centroid_points_old,centroid_points,0.01)): # python function
        break
print "Centroids\n"
print centroid_points


# # Visualize the results

# In[10]:

pylab.plot(samples1[:, 0], samples1[:, 1],'*', color = 'red')
pylab.plot(samples2[:, 0], samples2[:, 1],'o',color = 'blue')
pylab.plot(samples3[:, 0], samples3[:, 1],'+',color = 'green')
for point in centroid_points:
    pylab.plot(point[0], point[1], '*',color='pink',markersize=20)
pylab.show()


# In[ ]:



