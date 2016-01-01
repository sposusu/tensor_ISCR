import sys
import os
import operator


idir = sys.argv[1]
odir = None
if len(sys.argv)>2:
    odir = sys.argv[2]

results = {}
thetas = {}
for i in range(1,11):
    
    fin = file(idir+'log.fold'+str(i))
    thetas[i] = {} 
    lines = fin.readlines()
    for j in range(len(lines)):
	line = lines[j]
	tokens = line.split('\t')
	if len(tokens)>=3:
	    params = tokens[0].replace('(','').replace(')','').split(',')
	    set = tokens[1]
	    n1err = float(tokens[-1])
	    weight = 17.0
	    params.append(set)
	    key = tuple(params)
	    if i==10:
		weight = 10.0
	    if results.has_key(key):
		results[key] += weight*n1err/163.0
	    else:
		results[key] = weight*n1err/163.0
	elif line.startswith('theta'):
	    tokens = lines[j-2].split('\t')
	    params = tokens[0].replace('(','').replace(')','').split(',')
	    set = tokens[1]
	    params.append(set)
	    key = tuple(params)
	    theta = [w for w in lines[j+1].split()]
	    if thetas[i].has_key(key):
		thetas[i][key].append(theta)
	    else:
		thetas[i][key] = [theta]

best = []
bestDev = 100

sortedRes = sorted(results.iteritems(),key=operator.itemgetter(0))

for i in range(0,len(sortedRes),3):
    
    item = sortedRes[i]
    print '%s\t%.4f\n%s\t%.4f\n%s\t%.4f' % (\
	    sortedRes[i][0],sortedRes[i][1],\
	    sortedRes[i+1][0],sortedRes[i+1][1],\
	    sortedRes[i+2][0],sortedRes[i+2][1])
    
    if bestDev>sortedRes[i][1]:
	best = [sortedRes[i][0],sortedRes[i][1],sortedRes[i+1][1],sortedRes[i+2][1]]
	bestDev = sortedRes[i][1]

    print '--------------------------------------'

print 'BestDev:',best[0]
print 'train:%.4f\ndev:%.4f\ntest:%.4f' % (best[3],best[1],best[2])

if not odir==None:
    for f in range(1,11):
	print odir+'theta.fold'+str(f)
	fout = file(odir+'theta.fold'+str(f),'w')
	for theta in thetas[f][best[0]]:
	    for w in theta:
		fout.write(w+' ')
	    fout.write('\n')
	fout.close()

