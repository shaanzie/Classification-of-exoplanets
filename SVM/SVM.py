#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class Support_Vector_Machine:

    def __init__(self, visualization=True):

        # Making some initialisations for the visuals

        self.visualization = visualization

        self.colors = {1:'r',-1:'b'}

        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)


    # Definition for the fit method goes here

    def fit(self, data):

        self.data = data
        options = {}

        converts = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        # These converts allow transforms on the data

        all_data = []
        for yi in self.data:

            for featuredict in self.data[yi]:

                for feature in featuredict:

                    all_data.append(feature)

        self.featuremax = max(all_data)

        self.min_feature_value = min(all_data)

        all_data = None
        
        # Here we take the incremental step sizes

        size_of_step = [self.featuremax * 0.1,
                      self.featuremax * 0.01,
                      self.featuremax * 0.001,
                      ]


        b_range_multiple = 2
        
        b_multiple = 5

        latest_optimum = self.featuremax*10
        
        for step in size_of_step:

            w = np.array([latest_optimum,latest_optimum])

            optimized = False

            # Optimizing the model based on weights

            while not optimized:

                for b in np.arange(-1*(self.featuremax*b_range_multiple),

                                   self.featuremax*b_range_multiple,

                                   step*b_multiple):

                    for transformation in converts:

                        w_t = w*transformation
                        found_option = True
                        
                        for i in self.data:

                            for xi in self.data[i]:

                                yi=i
                                if not yi*(np.dot(w_t,xi[:int(w_t.shape[0])])+b) >= 1:

                                    found_option = False
                                    
                        if found_option:
                            options[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    
                    optimized = True
                    print('Step optimzized')
                else:
                    w = w - step

            norms = sorted([n for n in options])
            opt_choice = options[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                print(xi,':',yi*(np.dot(self.w.reshape(xi.shape),xi)+self.b))     


    # Use this for prediction        

    def predict(self,features):

        classification = np.sign(np.dot(np.array(features),self.w)+self.b)

        if classification !=0 and self.visualization:

            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])

        return classification

    def visualize(self):

        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
        def plane(x,w,b,v):
            
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.featuremax*1.1)
        optimal_x_min = datarange[0]
        optimal_x_max = datarange[1]

        psv1 = plane(optimal_x_min, self.w, self.b, 1)
        psv2 = plane(optimal_x_max, self.w, self.b, 1)
        self.ax.plot([optimal_x_min,optimal_x_max],[psv1,psv2], 'k')


        nsv1 = plane(optimal_x_min, self.w, self.b, -1)
        nsv2 = plane(optimal_x_max, self.w, self.b, -1)
        self.ax.plot([optimal_x_min,optimal_x_max],[nsv1,nsv2], 'k')


        db1 = plane(optimal_x_min, self.w, self.b, 0)
        db2 = plane(optimal_x_max, self.w, self.b, 0)
        self.ax.plot([optimal_x_min,optimal_x_max],[db1,db2], 'y--')

        plt.show()


df = pd.read_csv('../new_data.csv')
df = df[['P. Habitable', 'P. Min Mass (EU)', 'P. Mass (EU)']]



data_dict = dict()
list1in = list()
list1 = list()
list0 = list()
list0in = list()
for i, j in df.iterrows(): 
    if(j[0] == 1):
        list1in.append(j[1])
        list1in.append(j[2])
    else:
        list0in.append(j[1])
        list0in.append(j[2])
    list1.append(list1in)
    list0.append(list0in)
    
data_dict = {1: np.array(list1), -1: np.array(list0)}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

for p in np.array(list0):
    svm.predict(p)





