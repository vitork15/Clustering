import numpy as np
import matplotlib as mpl
import random

'''
K-Means clustering algorithm implementation with the use of the gaussian kernel.
This code offers both the hard clustering and the fuzzy clustering versions of the algorithm.
'''

def euclidian_distance(value1, value2):
    distance = 0
    for i in range(len(value1)):
        distance += (value1[i]-value2[i])**2
    return distance 

class gaussiankernelKM:
    
    def __init__(self):
        self.prototype_vector = []
        self.global_prototype = None
        self.membership_vector = []
        self.width_vector = []
        self.dataset = []
        self.cluster_num = 0
        self.adequacy_history = [] # Saves the history of the adequacy criterion over the steps

    def kernel(self, value, prototype, width_term):
        return np.exp(-euclidian_distance(value, prototype)/width_term)
    
    def estimate_width_vector(self):
        self.width_vector = [0 for i in range(len(self.dataset))]
        for i in range(len(self.dataset)):
            distance_vector = []
            for k in range(len(self.dataset)):
                if k != i: distance_vector.append(euclidian_distance(self.dataset[i], self.dataset[k]))
            self.width_vector[i] = (np.quantile(distance_vector,0.1)+np.quantile(distance_vector,0.9))/2

    def calculate_adequacy(self):
        data_num = len(self.dataset)
        adequacy = 0
        for k in range(self.cluster_num):
            for i in range(data_num):
                adequacy += self.membership_vector[i][k]*2*(1-self.kernel(self.dataset[i],self.prototype_vector[k],self.width_vector[i]))
        return adequacy

    def hard_cluster(self, dataset, cluster_num):

        # Initialization

        self.dataset = dataset
        self.cluster_num = cluster_num
        feature_num = len(dataset[0]) # Supposes the dataset is set up correctly
        data_num = len(dataset)
        self.prototype_vector = [[0 for col in range(feature_num)] for row in range(cluster_num)]
        self.global_prototype = [0 for col in range(feature_num)]
        self.membership_vector = [[0 for col in range(cluster_num)] for row in range(data_num)]
        self.adequacy_history = []

        self.estimate_width_vector()

        t = 0

        starting_prototypes = random.sample(range(data_num), cluster_num)
        for i in starting_prototypes: self.prototype_vector.append(dataset[i])
        self.global_prototype = dataset[random.choice(range(data_num))]

        # Calculate membership of each data point for the first time

        for i in range(data_num):
            temp_cluster = 0
            max_distance = np.inf
            for k in range(cluster_num):
                distance = 2*(1-self.kernel(self.dataset[i], self.prototype_vector[k], self.width_vector[i]))
                if(distance<max_distance): 
                    temp_cluster = i
                    max_distance = distance
            for k in range(cluster_num):
                if(i == temp_cluster): self.membership_vector[i][k] = 1
                else: self.membership_vector[i][k] = 0

        self.adequacy_history.append(self.calculate_adequacy()) # Save adequacy criterion on t = 0
        
        while(True):
            # First iterative step: Compute new prototypes based on current data points membership
            t = t + 1
            print("Step ", t)

            for k in range(self.cluster_num):
                for j in range(feature_num):
                    total_sum = 0
                    weight_sum = 0 
                    for i in range(data_num):
                        total_sum += self.membership_vector[i][j]*self.kernel(self.dataset[i], self.prototype_vector[k], self.width_vector[i])*self.dataset[i][j]
                        weight_sum += self.membership_vector[i][j]*self.kernel(self.dataset[i], self.prototype_vector[k], self.width_vector[i])
                    self.prototype_vector[k][j] = total_sum/weight_sum

            for j in range(feature_num):
                total_sum = 0
                weight_sum = 0 
                for i in range(data_num):
                    total_sum += self.membership_vector[i][j]*self.kernel(self.dataset[i], self.global_prototype, self.width_vector[i])*self.dataset[i][j]
                    weight_sum += self.membership_vector[i][j]*self.kernel(self.dataset[i], self.global_prototype, self.width_vector[i])
                self.global_prototype[j] = total_sum/weight_sum

            # Second iterative step: Modify data points membership based on the new prototypes

            test = False
            for i in range(data_num):
                membership = [k for k, v in enumerate(self.membership_vector[i]) if v == 1]
                winning_cluster = 0
                max_distance = np.inf
                for k in range(cluster_num):
                    distance = 2*(1-self.kernel(self.dataset[i], self.prototype_vector[k], self.width_vector[i]))
                    if(distance<max_distance): 
                        winning_cluster = i
                        max_distance = distance
                if(winning_cluster != membership):
                    test = True
                    for k in range(cluster_num):
                        if(i == winning_cluster): self.membership_vector[i][k] = 1
                        else: self.membership_vector[i][k] = 0

            self.adequacy_history.append(self.calculate_adequacy()) # Save adequacy criterion on current step

            if(test == False): break # If none of the data points get modified, stop the algorithm
                


    def fuzzy_cluster(self, dataset, cluster_num):
        NotImplemented



def main():
    fixed_test_dataset = [[1,2],[3.4,5.4],[11,22],[11.4,25.4],[10,20],[4.4,6.4],[1.9,-1.7]]
    model = gaussiankernelKM()
    model.hard_cluster(fixed_test_dataset, 2)
    print(model.adequacy_history)

if __name__ == "__main__":
    main()