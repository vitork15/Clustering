import numpy as np
import sklearn.datasets as sk
#import matplotlib as mpl
import random

'''
K-Means clustering algorithm implementation with the use of kernels.
This code offers both the hard clustering and the fuzzy clustering versions of the algorithm using the gaussian kernel.
'''

def estimate_width_term(dataset):
        distance_vector = []
        for i in range(len(dataset)):
            for k in range(len(dataset)):
                if k != i: distance_vector.append(euclidian_distance(dataset[i], dataset[k]))
        width = (np.quantile(distance_vector,0.1)+np.quantile(distance_vector,0.9))/2
        return width

def euclidian_distance(value1, value2):
    distance = 0
    for i in range(len(value1)):
        distance += (value1[i]-value2[i])**2
    return distance 

def gaussian_kernel(value, prototype, width_term):
    return np.exp(-euclidian_distance(value, prototype)/width_term)

class kernelKM:
    
    def __init__(self, kernel, *kernel_args):
        self.prototype_vector = []
        self.global_prototype = None
        self.membership_vector = []
        self.kernel = kernel
        self.kernel_args = kernel_args
        self.dataset = []
        self.cluster_num = 0
        self.adequacy_history = [] # Saves the history of the adequacy criterion over the steps

    def kernel(self, value, prototype, width_term):
        return np.exp(-euclidian_distance(value, prototype)/width_term)
    
    def estimate_width_vector(self):
        self.width_vector = [0 for i in range(len(self.dataset))]
        distance_vector = []
        for i in range(len(self.dataset)):
            for k in range(len(self.dataset)):
                if k != i: distance_vector.append(euclidian_distance(self.dataset[i], self.dataset[k]))
        distance_vector.sort()
        for i in range(len(self.dataset)):
            self.width_vector[i] = (np.quantile(distance_vector,0.1)+np.quantile(distance_vector,0.9))/2

    def calculate_adequacy(self, fuzzifier=1):
        data_num = len(self.dataset)
        adequacy = 0
        for k in range(self.cluster_num):
            for i in range(data_num):
                adequacy += (self.membership_vector[i][k]**fuzzifier)*2*(1-self.kernel(self.dataset[i],self.prototype_vector[k],self.width_vector[i]))
        return adequacy
    
    def fuzzy_to_hard(self): # Converts a fuzzy membership matrix into a discrete one for interpretation purposes
        hard_membership_vector = [[0 for col in range(self.cluster_num)] for row in range(len(self.dataset))]
        for i in range(len(self.membership_vector)):
            k = np.argmax(self.membership_vector[i])
            hard_membership_vector[i][k] = 1
        return hard_membership_vector

    def hard_cluster(self, dataset, cluster_num):

        random.seed()

        # Initialization

        self.dataset = dataset
        self.cluster_num = cluster_num
        feature_num = len(dataset[0]) # Supposes the dataset is set up correctly
        data_num = len(dataset)
        self.prototype_vector = []
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
                    temp_cluster = k
                    max_distance = distance
            for k in range(cluster_num):
                if(k == temp_cluster): self.membership_vector[i][k] = 1
                if(k == temp_cluster): self.membership_vector[i][k] = 1
                else: self.membership_vector[i][k] = 0

        self.adequacy_history.append(self.calculate_adequacy()) # Save adequacy criterion on t = 0
        
        while(True):
            # First iterative step: Compute new prototypes based on current data points membership
            t = t + 1

            for k in range(self.cluster_num):
                for j in range(feature_num):
                    total_sum = 0
                    weight_sum = 0 
                    for i in range(data_num):
                        total_sum += self.membership_vector[i][k]*self.kernel(self.dataset[i], self.prototype_vector[k], *self.kernel_args)*self.dataset[i][j]
                        weight_sum += self.membership_vector[i][k]*self.kernel(self.dataset[i], self.prototype_vector[k], *self.kernel_args)
                    self.prototype_vector[k][j] = total_sum/weight_sum
            for k in range(self.cluster_num):
                for j in range(feature_num):
                    total_sum = 0
                    weight_sum = 0 
                    for i in range(data_num):
                        total_sum += self.membership_vector[i][k]*self.kernel(self.dataset[i], self.global_prototype, *self.kernel_args)*self.dataset[i][j]
                        weight_sum += self.membership_vector[i][k]*self.kernel(self.dataset[i], self.global_prototype, *self.kernel_args)
                    self.global_prototype[j] = total_sum/weight_sum

            # Second iterative step: Modify data points membership based on the new prototypes

            test = False
            for i in range(data_num):
                membership = [k for k, v in enumerate(self.membership_vector[i]) if v == 1][0]
                winning_cluster = 0
                max_distance = np.inf
                for k in range(cluster_num):
                    distance = 2*(1-self.kernel(self.dataset[i], self.prototype_vector[k], *self.kernel_args))
                    if(distance<max_distance): 
                        winning_cluster = k
                        max_distance = distance
                if(winning_cluster != membership):
                    test = True
                    self.membership_vector[i][winning_cluster] = 1
                    self.membership_vector[i][membership] = 0

            self.adequacy_history.append(self.calculate_adequacy()) # Save adequacy criterion on current step

            if(test == False): break # If none of the data points get modified, stop the algorithm
                
    def hard_cluster_with_reinitialization(self, dataset, cluster_num, reinitializations=5):
        self.hard_cluster(dataset, cluster_num)
        current_adequacy = self.adequacy_history[-1]
        current_solution = (self.prototype_vector, self.membership_vector, self.global_prototype, self.adequacy_history)
        while reinitializations > 0:
            self.hard_cluster(dataset, cluster_num)
            if(current_adequacy > self.adequacy_history[-1]):
                current_adequacy = self.adequacy_history[-1]
                current_solution = (self.prototype_vector, self.membership_vector, self.global_prototype, self.adequacy_history)
            reinitializations = reinitializations - 1
        self.prototype_vector, self.membership_vector, self.global_prototype, self.adequacy_history = current_solution


    def fuzzy_cluster(self, dataset, cluster_num, fuzzifier=2, max_iterations=50, epsilon=1e-12):
        
        random.seed()

        # Initialization

        self.dataset = dataset
        self.cluster_num = cluster_num
        feature_num = len(dataset[0]) # Supposes the dataset is set up correctly
        data_num = len(dataset)
        self.prototype_vector = []
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
            distance_to_k = []
            for k in range(cluster_num):
                distance_to_k.append(1-self.kernel(self.dataset[i], self.prototype_vector[k], self.width_vector[i]))
            zeros = [k for k, v in enumerate(distance_to_k) if v == 0]
            if(len(zeros) == 0):
                for k in range(cluster_num):
                    self.membership_vector[i][k] = 0
                    for h in range(cluster_num):
                        self.membership_vector[i][k] += (distance_to_k[k]/distance_to_k[h])**(1/(fuzzifier-1))
                    self.membership_vector[i][k] = self.membership_vector[i][k]**(-1)
            else:
                for k in zeros:
                    self.membership_vector[i][k] = 1/len(zeros)

        self.adequacy_history.append(self.calculate_adequacy(fuzzifier)) # Save adequacy criterion on t = 0
        
        while(True):
            # First iterative step: Compute new prototypes based on current data points membership
            t = t + 1

            for k in range(self.cluster_num):
                for j in range(feature_num):
                    total_sum = 0
                    weight_sum = 0 
                    for i in range(data_num):
                        total_sum += (self.membership_vector[i][k]**fuzzifier)*self.kernel(self.dataset[i], self.prototype_vector[k], *self.kernel_args)*self.dataset[i][j]
                        weight_sum += (self.membership_vector[i][k]**fuzzifier)*self.kernel(self.dataset[i], self.prototype_vector[k], *self.kernel_args)
                    self.prototype_vector[k][j] = total_sum/weight_sum
            for k in range(self.cluster_num):
                for j in range(feature_num):
                    total_sum = 0
                    weight_sum = 0 
                    for i in range(data_num):
                        total_sum += (self.membership_vector[i][k]**fuzzifier)*self.kernel(self.dataset[i], self.global_prototype, *self.kernel_args)*self.dataset[i][j]
                        weight_sum += (self.membership_vector[i][k]**fuzzifier)*self.kernel(self.dataset[i], self.global_prototype, *self.kernel_args)
                    self.global_prototype[j] = total_sum/weight_sum

            # Second iterative step: Modify data points membership based on the new prototypes

            for i in range(data_num):
                distance_to_k = []
                for k in range(cluster_num):
                    distance_to_k.append(1-self.kernel(self.dataset[i], self.prototype_vector[k], self.width_vector[i]))
                zeros = [k for k, v in enumerate(distance_to_k) if v == 0]
                if(len(zeros) == 0):
                    for k in range(cluster_num):
                        self.membership_vector[i][k] = 0
                        for h in range(cluster_num):
                            self.membership_vector[i][k] += (distance_to_k[k]/distance_to_k[h])**(1/(fuzzifier-1))
                        self.membership_vector[i][k] = self.membership_vector[i][k]**(-1)
                else:
                    for k in range(cluster_num):
                        self.membership_vector[i][k] = 0
                    for k in zeros:
                        self.membership_vector[i][k] = 1/len(zeros)

            self.adequacy_history.append(self.calculate_adequacy(fuzzifier)) # Save adequacy criterion on current step

            if(abs(self.adequacy_history[-1] - self.adequacy_history[-2]) < epsilon or t > max_iterations): break # Stopping condition

    def fuzzy_cluster_with_reinitialization(self, dataset, cluster_num, reinitializations=5, fuzzifier=2, max_iterations=50, epsilon=1e-12):
        self.fuzzy_cluster(dataset, cluster_num, fuzzifier, max_iterations, epsilon)
        current_adequacy = self.adequacy_history[-1]
        current_solution = (self.prototype_vector, self.membership_vector, self.global_prototype, self.adequacy_history)
        while reinitializations > 0:
            self.fuzzy_cluster(dataset, cluster_num, fuzzifier, max_iterations, epsilon)
            if(current_adequacy > self.adequacy_history[-1]):
                current_adequacy = self.adequacy_history[-1]
                current_solution = (self.prototype_vector, self.membership_vector, self.global_prototype, self.adequacy_history)
            reinitializations = reinitializations - 1
        self.prototype_vector, self.membership_vector, self.global_prototype, self.adequacy_history = current_solution



def hard_clustering_test(test_dataset, classes):
    model = kernelKM(gaussian_kernel, estimate_width_term(test_dataset))
    model.hard_cluster_with_reinitialization(test_dataset, 3)
    class_elements = [[] for i in range(model.cluster_num)]
    true_classes = [[] for i in range(model.cluster_num)]
    for k in range(1,len(model.adequacy_history)):
        if(model.adequacy_history[k-1]>model.adequacy_history[k]):
            print(f"ITERAÇÃO {k} - J: {model.adequacy_history[k]}, DIMINUIU")
        else:
            print(f"ITERAÇÃO {k} - J: {model.adequacy_history[k]}, AUMENTOU")
    print("")
    for i in range(len(model.prototype_vector)):
        print(model.prototype_vector[i])
    print("")
    for i in range(len(classes)):
        true_classes[classes[i]].append(i)
    for i in range(len(model.membership_vector)):
        class_elements[[n for n, m in enumerate(model.membership_vector[i]) if m==1][0]].append(i)
    for i in range(model.cluster_num):
        print(f"CLASS {i} ELEMENTS: {class_elements[i]}")
    print("")
    for i in range(model.cluster_num):
        print(f"TRUE CLASS {i} ELEMENTS: {true_classes[i]}")

def fuzzy_clustering_test(test_dataset, classes):
    model = kernelKM(gaussian_kernel, estimate_width_term(test_dataset))
    model.fuzzy_cluster_with_reinitialization(test_dataset, 3, fuzzifier=1.5)
    hard_classes = model.fuzzy_to_hard()
    class_elements = [[] for i in range(model.cluster_num)]
    true_classes = [[] for i in range(model.cluster_num)]
    for k in range(1,len(model.adequacy_history)):
        if(model.adequacy_history[k-1]>model.adequacy_history[k]):
            print(f"ITERAÇÃO {k} - J: {model.adequacy_history[k]}, DIMINUIU")
        else:
            print(f"ITERAÇÃO {k} - J: {model.adequacy_history[k]}, AUMENTOU")
    print("")
    for i in range(len(model.prototype_vector)):
        print(model.prototype_vector[i])
    print("")
    for i in range(len(classes)):
        true_classes[classes[i]].append(i)
    for i in range(len(hard_classes)):
        class_elements[[n for n, m in enumerate(hard_classes[i]) if m==1][0]].append(i)
    for i in range(model.cluster_num):
        print(f"CLASS {i} ELEMENTS: {class_elements[i]}")
    print("")
    for i in range(model.cluster_num):
        print(f"TRUE CLASS {i} ELEMENTS: {true_classes[i]}")    

def main():
    iris = sk.load_iris()
    test_dataset = iris['data']
    classes = iris['target']
    fuzzy_clustering_test(test_dataset, classes)

if __name__ == "__main__":
    main()