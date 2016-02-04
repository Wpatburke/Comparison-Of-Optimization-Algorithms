# William Burke
# Comparison of Various Optimization Algorithms. 
# Last Updated December 15th
# 
# Algorithms compared: 
# > Proximal Gradient Descent
# > Accelerated Proximal Gradient Descent
# > Coordinate Descent
# > Alternating Direction Method of Multipliers (ADMM)
# 


import numpy as np
import matplotlib.pyplot as plt
import math
import time


#Solution Vector Size
p = 100 #sys.argv[1] 
n = 100
lambd = np.sqrt(2*n*np.log(p))


x = np.zeros(p)
A = np.zeros((n,p))
b = np.zeros(n)
eta = np.random.normal(0,1,n)


for j in range(20):
	index = np.random.randint(0,p)
	x[index] = np.random.normal(0,10)

for i in range(len(A)):
	for j in range(len(A[0])):
		A[i,j] = np.random.normal(0,1)



b = np.dot(A,x) + eta


def calculateRSME(sampleMatrix, predictionMatrix):
	sum = 0
	m = len(sampleMatrix)
	for i in range(m):
			sum = sum + (predictionMatrix[i] - sampleMatrix[i])**2
	return math.sqrt( sum / float(m))

# Proximal Gradien Descent

def L2_Norm(x):
	return np.sqrt(sum(abs(x)**2))
#print L2_Norm(np.dot(np.transpose(A),A))

gamma = 1/np.linalg.norm(np.dot(np.transpose(A),A))


def ObjectiveFunction(X,A,b, lambd):
	return .5 * np.linalg.norm(np.dot(A,X) - b)**2 + lambd*np.linalg.norm(X,ord = 1)


def SoftThresholdingOperator(B,lambd,gamma):
	S = np.copy(B)
	# if len(B) == 1:
	# 	if B > lambd*gamma:
	# 		S = B - lambd*gamma
	# 	elif B < -lambd*gamma: 
	# 		S = B + lambd*gamma
	# 	else:
	# 		S = 0
	# 	return S
	for i in range(len(B)):
		if B[i] > lambd*gamma:
			S[i] = B[i] - lambd*gamma
		elif B[i] < -lambd*gamma: 
			S[i] = B[i] + lambd*gamma
		else:
			S[i] = 0
	return S

#T = SoftThresholdingOperator(b,1)
# for i in range(len(T)):
# 	print b[i]
# 	print T[i]


k = 0
K = 10000


########################################################################
#Proximal Gradient Descent
########################################################################

start = time.time()
# for i in range(len(x)):
# 	print x[i]
# for Decreasing step
print("########################################################################")
print("########################################################################")
alpha = .4
t = 1
# print "lambda: ", lambd
# print "gamma: ",gamma
X =  np.zeros(p)#np.random.normal(0,1,p)# # Zero better
PGD_Plot=[]
while k < K:
	#t = t * alpha
	PGD_Plot.append(ObjectiveFunction(X,A,b,lambd))
	X = SoftThresholdingOperator(X + 
		gamma*np.dot(np.transpose(A),(b-np.dot(A,X))),lambd,gamma)
	k = k + 1
	
print "PGD RSME: ", calculateRSME(X,x)

# for i in range(len(x)):
# 	print abs(X[i] - x[i])

# print len(PGD_Plot)
# print PGD_Plot[50]


# for i in range(len(x)):
# 	print X[i]
# T = X.copy()
end = time.time()
print "runtime:", end - start

########################################################################
#Accelerated Proximal Gradient Descent

start = time.time()
k = 0
X =  np.zeros(p)#np.random.normal(0,1,p)# # Zero better
APGD_Plot=[]
while k < K-1:
	#t = t * alpha
	if k == 0 : # or k == 1 
		k = k + 1
		APGD_Plot.append(ObjectiveFunction(X,A,b,lambd))
		# Xold = X
		# Xcurrent = X
		# v = Xcurrent + (k-2)/(k+1)*(Xcurrent - Xold)
		X_k2 = X.copy()
		X = SoftThresholdingOperator(X + gamma*np.dot(np.transpose(A),(b-np.dot(A,X))),lambd,gamma)
		
		

	else:
		# k - 2
		k = k + 1
		APGD_Plot.append(ObjectiveFunction(X,A,b,lambd))
		X_k1 = X.copy()

		v = X_k1 + (k-2)/(k+1)*(X_k1 - X_k2)


		X = SoftThresholdingOperator(v + gamma*np.dot(np.transpose(A),(b-np.dot(A,v))),lambd,gamma)

		X_k2 = X_k1.copy()
		
		
print "APGD RSME: ", calculateRSME(X,x)

end = time.time()
print "runtime:", end - start
# for i in range(len(x)):
# 	print X[i]

# IterationRange = np.arange(0,K)
# # print len(IterationRange)
# # print len(PGD_Plot)
# # print len(APGD_Plot)

# # fig =plt.figure()
# # ax = fig.add_subplot(111)
# # ax.plot(IterationRange,PGD_Plot,c='b', label='PGD')
# # ax.plot(IterationRange,APGD_Plot,c='r', label='APGD')

# # plt.yscale('log')
# # plt.xscale('log')
# # plt.xlim(0,10000)
# # plt.legend()
# # plt.show()


########################################################################
#Coordinate Descent
########################################################################


start = time.time()

k = 0
X =  np.zeros(p)#np.random.normal(0,1,p)# # Zero better
CD_Plot=[]

r = b - np.dot(A,X)

N = 10000
APGD_Plot.append(ObjectiveFunction(X,A,b,lambd))
while k < N:
	CD_Plot.append(ObjectiveFunction(X,A,b,lambd))
	for i in range(len(x)):
		X_k1 = X[i]
		gamma = float(1/np.linalg.norm(A[:,i])**2)
		S = (np.dot(np.transpose(A[:,i]),r))*gamma + X[i]
		if S > lambd*gamma:
			S = S - lambd*gamma
		elif S < -lambd*gamma: 
			S = S + lambd*gamma
		else:
			S = 0
		X[i] = S #SoftThresholdingOperator( (np.transpose(A[:,i])*r)*gamma + X[i], lambd,gamma)
		r = r - A[:,i]*(X[i]-X_k1)
		
	k = k + 1
	


print"CD RSME:", calculateRSME(X,x)

end = time.time()
print "runtime:", end - start
# for i in range(len(x)):
# 	print X[i]

# IterationRange = np.arange(1,K+1)
# print len(IterationRange)
# print len(PGD_Plot)
# print len(APGD_Plot)


# fig =plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(IterationRange,PGD_Plot,c='b', label='PGD')
# ax.plot(IterationRange,APGD_Plot,c='r', label='APGD')
# ax.plot(IterationRange,CD_Plot,c='g',label='CD')

# plt.yscale('log')
# plt.xscale('log')
# plt.xlim(0,10000)
# plt.legend()
# plt.show()



# ########################################################################
# #Alternating Direction Method of Multipliers (ADMM)
# ########################################################################
start = time.time()
k = 0
X =  np.zeros(p)
ADMM_Plot = []
gamma = 15
y =  np.zeros(p)
z =  np.zeros(p)
#print z
while k < K:
	ADMM_Plot.append(ObjectiveFunction(X,A,b,lambd))
	X = np.dot(np.linalg.pinv(np.dot(A.T,A) + 
		np.identity(n)*gamma),(np.dot(A.T,b) + gamma*z - y))
	#print X.shape
	z = X + y/gamma
	for i in range(len(z)):
		#print z[i]
		if z[i] > lambd/gamma:
			z[i] = z[i] - lambd/gamma
		elif z[i] < -lambd/gamma: 
			z[i] = z[i] + lambd/gamma
		else:
			z[i] = 0


	
	y = y + gamma*(x - z)
	k = k + 1 

# for i in range(len(x)):
# 	print X[i]

print "ADMM RSME:",calculateRSME(X,x)

end = time.time()
print "runtime:", end - start

IterationRange = np.arange(1,K+1)
# print len(IterationRange)
# print len(PGD_Plot)
# print len(APGD_Plot)


fig =plt.figure()
ax = fig.add_subplot(111)
ax.plot(IterationRange,PGD_Plot,c='b', label='PGD')
ax.plot(IterationRange,APGD_Plot,c='r', label='APGD')
ax.plot(IterationRange,CD_Plot,c='g',label='CD')
ax.plot(IterationRange,ADMM_Plot,c='y',label='ADMM')

plt.yscale('log')
plt.xscale('log')
plt.xlim(0,10000)
plt.legend()
plt.show()

















