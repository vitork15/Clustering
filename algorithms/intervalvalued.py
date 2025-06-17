import numpy as np
import sklearn.datasets as sk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, adjusted_rand_score
import random
import scipy.io
import itertools
from utils.metrics import *


class AIFCM_ER:

    def __init__(self):
        self.prototype_vector = []
        #self.global_prototype = None
        self.membership_matrix = []
        self.weight_matrix = []
        self.data = []
        self.cluster_num = 0
        self.adequacy_history = [] # Saves the history of the adequacy criterion over the steps
        self.T_u = 1

    def objective_function(self):

        value = 0

        #distance component
        lower_components = ((self.data[:,np.newaxis,:,0]-self.prototype_vector[np.newaxis,:,:,0])**2)*self.lower_weight_matrix[np.newaxis,:,:]
        upper_components = ((self.data[:,np.newaxis,:,1]-self.prototype_vector[np.newaxis,:,:,1])**2)*self.upper_weight_matrix[np.newaxis,:,:]
        value += np.sum(self.membership_matrix * np.sum(lower_components + upper_components, axis=2))

        #entropy component
        not_zero = self.membership_matrix > 0
        value += self.T_u*np.sum(self.membership_matrix[not_zero]*np.log(self.membership_matrix[not_zero])) #xlogx is approx 0 at x=0

        return value

    def run(self, data, cluster_num, T_u=1, max_iterations=200, threshold=1e-10):
        random.seed()

        # Initialization

        self.data = data
        self.cluster_num = cluster_num
        feature_num = data.shape[1] # Supposes the dataset is set up correctly
        data_num = data.shape[0]
        self.prototype_vector = []
        self.upper_weight_matrix = np.ones((cluster_num,feature_num)) #[[1/p for row in range(feature_num)] for col in range(cluster_num)]
        self.lower_weight_matrix = np.ones((cluster_num,feature_num))
        self.membership_matrix = np.zeros((data_num, cluster_num))#[[0 for col in range(cluster_num)] for row in range(data_num)]
        self.adequacy_history = []
        self.T_u = T_u

        t = 0

        starting_prototypes = random.sample(range(data_num), cluster_num)
        self.prototype_vector = data[starting_prototypes].copy()

        starting_membership = random.choices(range(cluster_num), k=data_num)
        for i, k in enumerate(starting_membership):
            self.membership_matrix[i][k] = 1

        self.adequacy_history.append(self.objective_function()) # initial value of J

        while(t < max_iterations):

            t = t + 1

            #fuzzy cluster prototype computation (representation step)

            weight_sums = np.sum(self.membership_matrix, axis=0) 
            valid_clusters = weight_sums > 1e-10  # check zero division
            if np.any(valid_clusters): # check if not empty
                for bound in range(2):
                    weighted_sum = self.membership_matrix.T @ data[:, :, bound] 
                    self.prototype_vector[valid_clusters, :, bound] = weighted_sum[valid_clusters] / weight_sums[valid_clusters, np.newaxis]
                
            #width parameter computation (weighting step), not implemented for testing
            
            lower_components = np.sum(((self.data[:,np.newaxis,:,0]-self.prototype_vector[np.newaxis,:,:,0])**2)*self.membership_matrix[:,:,np.newaxis], axis=0)
            upper_components = np.sum(((self.data[:,np.newaxis,:,1]-self.prototype_vector[np.newaxis,:,:,1])**2)*self.membership_matrix[:,:,np.newaxis], axis=0)
            numerator = (np.prod(lower_components, axis=1)*np.prod(upper_components, axis=1))**(1/(2*feature_num))

            np.divide(numerator[:, np.newaxis], lower_components, out=self.lower_weight_matrix, where=lower_components > 1e-10)
            np.divide(numerator[:, np.newaxis], upper_components, out=self.upper_weight_matrix, where=upper_components > 1e-10)
            
            #membership degree computation (allocation step)
            
            lower_components = ((self.data[:,np.newaxis,:,0]-self.prototype_vector[np.newaxis,:,:,0])**2)*self.lower_weight_matrix[np.newaxis,:,:]
            upper_components = ((self.data[:,np.newaxis,:,1]-self.prototype_vector[np.newaxis,:,:,1])**2)*self.upper_weight_matrix[np.newaxis,:,:]
            distance = np.sum(lower_components + upper_components, axis=2)
            
            intermediate = np.exp(-distance/self.T_u)
            intermediate_sum = np.sum(intermediate, axis=1)
            valid_data = intermediate_sum > 1e-10
            
            self.membership_matrix[valid_data] = intermediate[valid_data]/intermediate_sum[valid_data,np.newaxis]

            #history + stop condition

            self.adequacy_history.append(self.objective_function())

            if(abs(self.adequacy_history[-1]-self.adequacy_history[-2]) < threshold):
                break

        return self

    def run_many(self, data, cluster_num, T_u=1, max_iterations=200, reinitializations = 30, threshold=1e-10):
        best_run = self.run(data, cluster_num, T_u, max_iterations, threshold)
        while(reinitializations > 0):
            trial_run = self.run(data, cluster_num, T_u, max_iterations, threshold)
            if(best_run.adequacy_history[-1] > trial_run.adequacy_history[-1]): best_run = trial_run
            reinitializations = reinitializations - 1

        return best_run

    def run_t(self, data, cluster_num, max_iterations=200, reinitializations = 30, interval = [0.05,1,0.05], threshold=1e-10):
        start, finish, jump = interval
        best_run = self.run_many(data, cluster_num, start, max_iterations, reinitializations, threshold)
        while(start < finish):
            start += jump
            trial_run = self.run_many(data, cluster_num, start, max_iterations, reinitializations, threshold)
            if(best_run.adequacy_history[-1] > trial_run.adequacy_history[-1]): best_run = trial_run

        return best_run

    def crispy_membership(self):
        '''
        Converts the fuzzy membership matrix U into a discrete one
        '''
        hard_membership_matrix = np.zeros((len(self.data), self.cluster_num))

        max_index = np.argmax(self.membership_matrix, axis=1)

        hard_membership_matrix[np.arange(len(max_index)), max_index] = 1

        return hard_membership_matrix

    def confusion_matrix(self, true_classes):

        true_classes_index = np.unique(true_classes)

        pred = true_classes_index[self.predict()]
        cm = confusion_matrix(true_classes, pred, labels=true_classes_index)

        return cm

    def predict(self):
        prediction = np.argmax(self.membership_matrix, axis=1)
        return prediction

    def analyze(self):
        print(f"J INICIAL: {self.adequacy_history[0]}")
        for k in range(1,len(self.adequacy_history)):
            if(self.adequacy_history[k-1]>self.adequacy_history[k]):
                print(f"ITERAÇÃO {k} - J: {self.adequacy_history[k-1]} -> {self.adequacy_history[k]}, DIMINUIU")
            elif(self.adequacy_history[k-1]==self.adequacy_history[k]):
                print(f"ITERAÇÃO {k} - J: {self.adequacy_history[k-1]} -> {self.adequacy_history[k]}, MANTEVE")
            else:
                print(f"ITERAÇÃO {k} - J: {self.adequacy_history[k-1]} -> {self.adequacy_history[k]}, AUMENTOU")

def loadintervaldotmat(filename):
    mat = scipy.io.loadmat(filename)

    raw_data = np.array(mat['data'])
    classes = raw_data[:, 0].astype(int)

    feature_data = raw_data[:, 1:]

    n_data, n_bounds = feature_data.shape
    n_features = n_bounds // 2
    structured_data = feature_data.reshape(n_data, n_features, 2)

    return structured_data, classes

def main():
    
    data, classes = loadintervaldotmat('../datasets/Car.mat')
    
    class_labels = np.unique(classes)
    model = AIFCM_ER()
    model = model.run_many(data, len(class_labels), T_u=2e4, reinitializations=500)

    model.analyze()

    print(f"RAND SCORE AJUSTADO: ", adjusted_rand_score(classes, class_labels[model.predict()]))
    print(f"RAND HULLERMEIER: ", rand_hullermeier(model.membership_matrix, classes, true_class=True))
    print(f"RAND FRIGUI: ", rand_frigui(model.membership_matrix, classes, true_class=True))

    cm = model.confusion_matrix(classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=range(model.cluster_num))
    disp.plot()
    plt.show()

    #print(data)

if __name__ == "__main__":
    main()