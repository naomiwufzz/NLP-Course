# Created by Liang at 2018/12/5

"""
Feature: 
# Basic machine learning steps

# Enter feature description here
	summary of basic steps for machine learning developement

Scenario: 
#Enter scenario name here
	assume the age of titanic's Passenger and the fare of ticket they bought has linear releationship

Test File Location: 
#Enter 

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


############################Data collection###########################
####using libariy such as numpy, pandas, tensorflow and so on
titanic_content = pd.read_csv(open('C:\\Users\\mathm\\Documents\\GitHub\\kaggle-titanic\\train.csv'))
titanic_content = titanic_content.dropna()


#####Data clean, similar as process of Data minning
age_with_fare = titanic_content[['age', 'fare']]
age_with_fare = age_with_fare[ (age_with_fare['age'] > 22) & (age_with_fare['fare'] < 400) &  (age_with_fare['fare'] > 130)]


age = np.array(age_with_fare['age'].tolist())
fare = np.array(age_with_fare['fare'].tolist())
############################Data collection###########################

#Lost function definition
def loss(simple_Y_value, model_Y_value): return np.mean(np.abs(simple_Y_value - model_Y_value))
#def loss(simple_Y_value, model_Y_value): return np.mean(np.abs(simple_Y_value - model_Y_value)**2)

#Model function definition
def model(x, a, b): return a * x + b
# def model(x,a,b,c): return a*x**2 +b*x + c

#Initialize parmas
a = 2
b = 2
### for model(two)
# c=3
min_loss = float('inf')

##### usually, it means how many samples used for each training####
batch = 0

##### up limit stop condition #####
total = 20000

model_Y_value = np.array([model(x, a, b) for x in age])
# model_Y_value = np.array([model(x, a, b, c) for x in age])

#one of the stop condition
eps = 10

#learning rate set
learning_rate = 1e-2

#machine learning algorithm, such as gradient or four directions try
# directions = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1),(-1, 1, -1), (-1,-1,1), (-1,-1,-1)]
directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]


while True:
	indices = np.random.choice(range(len(age)),size=20)
	
	sample_x = age[indices]
	sample_y = fare[indices]
	new_a,new_b = a,b
	# new_a,new_b,new_c = a,b,c

	model_Y_value=np.array([model(x,new_a,new_b) for x in sample_x])
	# model_Y_value=np.array([model(x,new_a,new_b,new_c) for x in sample_x])
	
	Error = loss(sample_y, model_Y_value)
	if Error < eps: 
		print('batch {} stopped, since loss {} low eps:{} while a={}  b={}'.format(batch, Error, eps,a,b))
		# print('batch {} stopped, since loss {} low eps:{} while a={}  b={} c={}'.format(batch, Error, eps,a,b,c))
		break

	for d in directions:
		da, db = d
		# da, db, dc = d

		if min_loss != float('inf'):
			_a = a + da * min_loss * learning_rate
			_b = b + db * min_loss * learning_rate
			# _a = a + da * min_loss * learning_rate
			# _b = b + db * min_loss * learning_rate
			# _c = c + dc * min_loss * learning_rate
		else:
			_a, _b = a + da, b + db
			# _a, _b, _c =a+da, b+db, c+dc

		model_Y_value = [model(x, _a, _b) for x in sample_x]
		# model_Y_value = [model(x, _a, _b, _c) for x in sample_x]
		Error = loss(sample_y, model_Y_value)

		if Error < min_loss:
			min_loss = Error
			new_a, new_b=_a, _b
			# new_a, new_b ,new_c = _a, _b, _c

	if batch % 100 == 0:
		print('batch {}/{} fare with {} *age + {}, with loss: {}'.format(batch, total, new_a, new_b, min_loss))
		# print('batch {}/{} fare with {} *age^2 + {} *age + {}, with loss: {}'.format(batch, total, new_a, new_b, new_c, min_loss))

	if batch > total: break

	batch += 1

	a, b= new_a,new_b
	# a, b, c= new_a, new_b, new_c

######plot#############
plt.scatter(age, fare)
plt.plot(age, [model(x, a, b) for x in age])
# plt.plot(age, [model(x, a, b, c) for x in age])
plt.show()

