import sys
import os
import operator


numofgauss = ''
var = ''
lamda = ''

if len(sys.argv)==5:
    numofgauss = '.' + sys.argv[2]
    var = '.' + sys.argv[3]
    lamda = '.'+sys.argv[4]

MAPs = []
Returns = []
#iters = [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000]
iters = [50,100,200,500,1000,2000,5000,10000]
for i in range(len(iters)):
    MAPs.append(0.0)
    Returns.append(0.0)


cnt = 0.0
for i in range(1,11,1):
    weight = 17.0
    if i==10:
	weight = 10.0
    fin = file(sys.argv[1]+'log.fold'+str(i)+numofgauss+var+lamda)
    for line in fin.readlines():
	tokens = line.replace('\n','').split()
	if len(tokens)<=0:
	    continue
	if tokens[0]=='Iteration':
	    cnt +=1
	    iter = int(tokens[1].replace(':',''))
	    if iter in iters:
		index = iters.index(iter)
		map = float(tokens[2].split(':')[1])
		MAPs[index]+=map*weight
		ret = float(tokens[-1])
		Returns[index]+=ret*weight

print 'finished folds:', cnt/float(len(iters))
for i in range(len(iters)):
    print iters[i],
print 

for i in range(len(MAPs)):
    print MAPs[i]/163.0,
print 

for i in range(len(Returns)):
    print Returns[i]/163.0,
print 



