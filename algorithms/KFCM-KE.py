import numpy as np
import sklearn.datasets as sk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, adjusted_rand_score
import random
import scipy.io
import itertools
from utils.metrics import *


class KFCM_KE:

    def __init__(self):
        self.prototype_vector = []
        #self.global_prototype = None
        self.membership_matrix = []
        self.width_matrix = []
        self.data = []
        self.cluster_num = 0
        self.adequacy_history = [] # Saves the history of the adequacy criterion over the steps
        self.T_u = 1
        self.fuzzifier = 1

    def gaussian(self, data_index, cluster_index):

        sum_components = ((self.data[data_index]-self.prototype_vector[cluster_index])**2)*self.width_matrix[cluster_index]
        gaussian = np.exp(-np.sum(sum_components, axis=2)/2)

        return gaussian

    def objective_function(self):

        value = 0

        #distance component
        sum_components = ((self.data[:,np.newaxis,:]-self.prototype_vector[np.newaxis,:,:])**2)*self.width_matrix[np.newaxis,:,:]
        gaussian = np.exp(-np.sum(sum_components, axis=2)/2)
        value += 2*np.sum((self.membership_matrix ** self.fuzzifier) * (1-gaussian))

        #entropy component
        width_plus_1 = 1+self.width_matrix
        value += self.T_w*np.sum(width_plus_1*np.log(width_plus_1))

        return value

    def run(self, data, cluster_num, T_w=1, fuzzifier=1, max_iterations=200, threshold=1e-10):
        random.seed()

        # Initialization

        self.data = data
        self.cluster_num = cluster_num
        feature_num = len(data[0]) # Supposes the dataset is set up correctly
        data_num = len(data)
        self.prototype_vector = []
        self.width_matrix = np.ones((cluster_num,feature_num))/feature_num #[[1/p for row in range(feature_num)] for col in range(cluster_num)]
        self.membership_matrix = np.zeros((data_num, cluster_num))#[[0 for col in range(cluster_num)] for row in range(data_num)]
        self.adequacy_history = []
        self.T_w = T_w
        self.fuzzifier = fuzzifier

        t = 0

        starting_prototypes = random.sample(range(data_num), cluster_num)
        for i in starting_prototypes:
            self.prototype_vector.append(data[i])
        self.prototype_vector = np.array(self.prototype_vector)

        starting_membership = random.choices(range(cluster_num), k=data_num)
        for i, k in enumerate(starting_membership):
            self.membership_matrix[i][k] = 1

        self.adequacy_history.append(self.objective_function()) # initial value of J

        while(t < max_iterations):
            
            #TODO Rename variables with better names 

            t = t + 1
            
            #width parameter computation (weighting step)
            sum_components = ((self.data[:,np.newaxis,:]-self.prototype_vector[np.newaxis,:,:])**2)*self.width_matrix[np.newaxis,:,:]
            gaussian = np.exp(-np.sum(sum_components, axis=2)/2)
            width_sum = np.sum(((self.membership_matrix ** self.fuzzifier) * gaussian)[:,:,np.newaxis] * (self.data[:,np.newaxis,:]-self.prototype_vector[np.newaxis,:,:])**2, axis=0)
            width_terms = np.exp(-(1/self.T_w)*width_sum)
            width_sum_terms = np.sum(width_terms, axis=1)
            
            for k in range(cluster_num):
                update_features = np.arange(feature_num)
                update = True
                while update:
                    update = False
                    for j in update_features:
                        if(j != -1):
                            width = (1 + len(update_features))*width_terms[k][j]/width_sum_terms[k]
                            if width <= 0:
                                update_features[j] = -1
                                update = True
                            else:
                                self.width_matrix[k][j] = width

            #fuzzy cluster prototype computation (representation step)
            sum_components = ((self.data[:,np.newaxis,:]-self.prototype_vector[np.newaxis,:,:])**2)*self.width_matrix[np.newaxis,:,:]
            gaussian = np.exp(-np.sum(sum_components, axis=2)/2)
            prototype_weights = (self.membership_matrix ** self.fuzzifier) * gaussian

            total_sum = prototype_weights.T @ data
            weight_sum = np.sum(prototype_weights, axis=0)[:, np.newaxis]

            self.prototype_vector = total_sum/weight_sum         

            #membership degree computation (allocation step)

            sum_components = ((self.data[:,np.newaxis,:]-self.prototype_vector[np.newaxis,:,:])**2)*self.width_matrix[np.newaxis,:,:]
            membership_terms = 1 - np.exp(-np.sum(sum_components, axis=2)/2)

            ratios = (membership_terms[:, :, np.newaxis]/membership_terms[:, np.newaxis, :])**(1 / (feature_num - 1))  # shape: (n_samples, cluster_num, cluster_num)
            denominator = np.sum(ratios, axis=2)  # shape: (n_samples, cluster_num)
            self.membership_matrix = 1/denominator

            #history + stop condition

            self.adequacy_history.append(self.objective_function())

            if(abs(self.adequacy_history[-1]-self.adequacy_history[-2]) < threshold):
                break

        return self

    def run_many(self, data, cluster_num, T_w=1, fuzzifier=1, max_iterations=200, reinitializations = 30, threshold=1e-10):
        best_run = self.run(data, cluster_num, T_w, fuzzifier, max_iterations, threshold)
        while(reinitializations > 0):
            trial_run = self.run(data, cluster_num, T_w, fuzzifier, max_iterations, threshold)
            if(best_run.adequacy_history[-1] > trial_run.adequacy_history[-1]): best_run = trial_run
            reinitializations = reinitializations - 1

        return best_run

    def run_t(self, data, cluster_num, max_iterations=50, fuzzifier=1, reinitializations = 30, interval = [0.05,1,0.05], threshold=1e-10):
        start, finish, jump = interval
        best_run = self.run_many(data, cluster_num, start, fuzzifier, max_iterations, reinitializations, threshold)
        while(start < finish):
            start += jump
            trial_run = self.run_many(data, cluster_num, start, fuzzifier, max_iterations, reinitializations, threshold)
            if(best_run.adequacy_history[-1] > trial_run.adequacy_history[-1]): best_run = trial_run

        return best_run

    def crispy_membership(self):
        '''
        Converts the fuzzy membership matrix U into a discrete one
        '''
        hard_membership_matrix = np.zeros((len(self.data), self.cluster_num))

        max_index = np.argmax(self.membership_matrix, 1)

        hard_membership_matrix[np.arange(len(max_index)), max_index] = 1

        return hard_membership_matrix

    def confusion_matrix(self, true_classes):

        true_classes_index = np.unique(true_classes)

        pred = true_classes_index[self.predict()]
        cm = confusion_matrix(true_classes, pred, labels=true_classes_index)

        return cm

    def predict(self):
        prediction = np.argmax(self.membership_matrix, 1)
        return prediction

    def analyze(self):
        print(f"J INICIAL: {self.adequacy_history[0]}")
        for k in range(1,len(self.adequacy_history)):
            if(self.adequacy_history[k-1]>self.adequacy_history[k]):
                print(f"ITERAÇÃO {k} - J: {self.adequacy_history[k-1]} -> {self.adequacy_history[k]}, DIMINUIU")
            else:
                print(f"ITERAÇÃO {k} - J: {self.adequacy_history[k-1]} -> {self.adequacy_history[k]}, AUMENTOU")

def loadintervaldotmat(filename):
    mat = scipy.io.loadmat(filename)

    structured_data = []
    classes = []

    for array in mat['data']:
        data_point = []

        for i in range(1,len(array),2):
            data_point.append(np.array([array[i],array[i+1]], dtype='float64'))

        structured_data.append(data_point)
        classes.append(int(array[0]))

    return np.array(structured_data), classes

def main():
    
    iris = sk.load_iris()
    data = np.array(iris['data'])
    classes = iris['target']
    
    class_labels = np.unique(classes)
    model = KFCM_KE()
    model = model.run_many(data, 3, T_w=40, fuzzifier=1.1, reinitializations=60)

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