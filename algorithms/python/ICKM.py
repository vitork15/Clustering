import numpy as np
import sklearn.datasets as sk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, adjusted_rand_score
import random
import scipy.io
import itertools
import timeit


class ICMKmodel:

    def __init__(self):
        self.prototype_vector = []
        #self.global_prototype = None
        self.membership_matrix = []
        self.lower_boundary_weights = [] # Weights for lower bounds
        self.upper_boundary_weights = [] # Weights for upper bounds
        self.data = []
        self.cluster_num = 0
        self.adequacy_history = [] # Saves the history of the adequacy criterion over the steps
        self.T_u = 1

    def interval_distance(self, interval_index, cluster_index):

        interval = self.data[interval_index]
        prototype = self.prototype_vector[cluster_index]

        distance = np.sum(self.lower_boundary_weights[cluster_index]*(interval[:,0]-prototype[:,0])**2 + self.upper_boundary_weights[cluster_index]*(interval[:,1]-prototype[:,1])**2)

        return distance
    
    def distance_matrix(self):
        matrix = np.empty((len(self.data),self.cluster_num))
        for k in range(self.cluster_num):
            matrix[k] = np.sum(self.lower_boundary_weights[k]*(self.data[:,:,0]-self.prototype_vector[k,:,0])**2 + self.upper_boundary_weights[k]*(self.data[:,:,0]-self.prototype_vector[k,:,1])**2, axis=0)
        return matrix

    def objective_function(self):
        value = 0

        for i in range(len(self.data)): # for each interval i
            for k in range(len(self.prototype_vector)): # for each prototype k

                #distance component
                for p in range(len(self.data[i])): # for each feature
                    value += self.membership_matrix[i][k]*self.interval_distance(i, k)
                #entropy component
                if(self.membership_matrix[i][k] > 1e-10): value += self.T_u*self.membership_matrix[i][k]*np.log(self.membership_matrix[i][k]) #approx x*logx = 0 at x = 0

        return value

    def run(self, data, cluster_num, T_u=1, max_iterations=50, threshold=1e-10):
        random.seed()

        # Initialization

        self.data = data
        self.cluster_num = cluster_num
        feature_num = len(data[0]) # Supposes the dataset is set up correctly
        data_num = len(data)
        self.prototype_vector = []
        self.lower_boundary_weights = np.ones((cluster_num,feature_num))#[[1 for row in range(feature_num)] for col in range(cluster_num)]
        self.upper_boundary_weights = np.ones((cluster_num,feature_num))#[[1 for row in range(feature_num)] for col in range(cluster_num)]
        self.membership_matrix = np.zeros((data_num, cluster_num))#[[0 for col in range(cluster_num)] for row in range(data_num)]
        self.adequacy_history = []
        self.T_u = T_u

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

            t = t + 1

            #representation step
            
            start_rep = timeit.default_timer()

            for k in range(cluster_num):
                if np.sum(self.membership_matrix[:,k]) > 1e-10: # check if it gets near zero
                    self.prototype_vector[k,:,0] = np.average(data[:,:,0],axis = 0,weights=self.membership_matrix[:,k]) # lower bound
                    self.prototype_vector[k,:,1] = np.average(data[:,:,1],axis = 0,weights=self.membership_matrix[:,k]) # upper bound

            end_rep = timeit.default_timer()
            start_weight = timeit.default_timer()
            
            #weighting step
            for k in range(cluster_num):
                lower_distance = self.membership_matrix[:,k]@(self.data[:,:,0]-self.prototype_vector[k,:,0])**2 # sum u_ik(a_ij-alpha_kj) = U_k*(A-alpha_k)
                upper_distance = self.membership_matrix[:,k]@(self.data[:,:,1]-self.prototype_vector[k,:,1])**2

                prod = np.prod(lower_distance**(1/(2*feature_num)))*np.prod(upper_distance**(1/(2*feature_num)))

                for j in range(feature_num):
                    # if division by zero is found, ignore until next step
                    if(lower_distance[j] > 1e-10): self.lower_boundary_weights[k][j] = prod/lower_distance[j]
                    if(upper_distance[j] > 1e-10): self.upper_boundary_weights[k][j] = prod/upper_distance[j]

            end_weight = timeit.default_timer()
            start_alloc = timeit.default_timer()
            
            #allocation step
            for i in range(data_num):
                denominator = 0
                for k in range(cluster_num):
                    denominator += np.exp(-self.interval_distance(i, k)/self.T_u)
                if denominator > 1e-10:
                  for k in range(cluster_num):
                      self.membership_matrix[i][k] = np.exp(-self.interval_distance(i, k)/self.T_u)/denominator
                      
            end_alloc = timeit.default_timer()
            
            print(f"REP: {end_rep-start_rep}, WEIGHT: {end_weight-start_weight}, ALLOC: {end_alloc-start_alloc}")

            self.adequacy_history.append(self.objective_function())

            if(abs(self.adequacy_history[-1]-self.adequacy_history[-2]) < threshold): break;

        return self

    def run_many(self, data, cluster_num, T_u=1, max_iterations=50, reinitializations = 5, threshold=1e-10):
        best_run = self.run(data, cluster_num, T_u, max_iterations, threshold)
        while(reinitializations > 0):
            trial_run = self.run(data, cluster_num, T_u, max_iterations, threshold)
            if(best_run.adequacy_history[-1] > trial_run.adequacy_history[-1]): best_run = trial_run
            reinitializations = reinitializations - 1

        return best_run

    def run_t(self, data, cluster_num, max_iterations=50, reinitializations = 5, interval = [0.05,1,0.05], threshold=1e-10):
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

        max_index = np.argmax(self.membership_matrix, 1)

        for num, cluster in enumerate(max_index):
            hard_membership_matrix[num][cluster] = 1

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

def loaddotmat(filename):
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

def rand_hullermeier(P, Q, true_class=False):
    '''
    Receives membership matrices P and Q to generate Hullemeier rand index using L¹ metric.
    The boolean flag true_class treats Q as a true class vector in a similar manner to true/prediction metrics.
    Source for implementation can be found on https://hal.science/hal-00734389v1/document.
    '''
    
    distance = 0
    cluster_num = len(P[0])
    data_num = len(P)
    
    if true_class is not True:
        for i, j in itertools.product(range(data_num), repeat=2):
            if i >= j:
                continue # 1 <= i < j <= n condition
            
            e_q, e_p = 0, 0
            
            for cluster in range(cluster_num):
                e_q += abs(Q[i][cluster]-Q[j][cluster])
                e_p += abs(P[i][cluster]-P[j][cluster])
            
            distance += abs(e_q - e_p)
            
        return 1 - 2*distance/(data_num*(data_num-1))

    else:
        for i, j in itertools.product(range(data_num), repeat=2):
            if i >= j:
                continue # 1 <= i < j <= n condition
            
            e_q, e_p = 0, 0
            
            e_q += 1 - int(Q[i]==Q[j])
            for cluster in range(cluster_num):
                e_p += abs(P[i][cluster]-P[j][cluster])
            
            distance += abs(e_q - e_p)
            
        return 1 - 2*distance/(data_num*(data_num-1))
    
def rand_frigui(P, Q, true_class=False):
    '''
    Receives membership matrices P and Q to generate Frigui rand index.
    The boolean flag true_class treats Q as a true class vector in a similar manner to true/prediction metrics.
    '''
    # these variables are similar to TP, FN, TN and FP
    same_same = 0
    same_different = 0
    different_different = 0
    different_same = 0
    
    cluster_num = len(P[0])
    data_num = len(P)
    
    if true_class is not True:
        for i, j in itertools.product(range(data_num), repeat=2):
            if i >= j:
                continue # 1 <= i < j <= n condition
            
            psi1, psi2 = 0, 0
            
            for cluster in range(cluster_num):
                psi1 += P[i][cluster]*P[j][cluster]
                psi2 += Q[i][cluster]*Q[j][cluster]
            
            same_same += psi1*psi2
            same_different += psi1*(1-psi2)
            different_same += (1-psi1)*psi2
            different_different += (1-psi1)*(1-psi2)
            
        return (same_same+different_different)/(same_same+same_different+different_same+different_different)

    else:
        for i, j in itertools.product(range(data_num), repeat=2):
            if i >= j:
                continue # 1 <= i < j <= n condition
            
            psi1, psi2 = 0, 0
            
            for cluster in range(cluster_num):
                psi1 += P[i][cluster]*P[j][cluster]
            
            psi2 = int(Q[i]==Q[j])
            
            same_same += psi1*psi2
            same_different += psi1*(1-psi2)
            different_same += (1-psi1)*psi2
            different_different += (1-psi1)*(1-psi2)
            
        return (same_same+different_different)/(same_same+same_different+different_same+different_different)

def main():

    data, classes = loaddotmat('Wine.mat')
    class_labels = np.unique(classes)
    model = ICMKmodel()
    #model = model.run_t(data, max(classes), max_iterations=300, reinitializations=30, interval=[1,100,0.5])
    model = model.run(data, max(classes), max_iterations=100, T_u = 0.1)

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