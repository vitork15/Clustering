import numpy as np
import sklearn.datasets as sk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, adjusted_rand_score
import random
import scipy.io
import gmpy2


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
        distance = 0
        interval = self.data[interval_index]
        prototype = self.prototype_vector[cluster_index]
        
        #if(len(interval) != len(prototype)): 
        #    raise ValueError("data and prototype don't have the same dimensions")
        
        for p in range(len(interval)): # for each feature
            distance += self.lower_boundary_weights[cluster_index][p]*(interval[p][0]-prototype[p][0])**2 + self.upper_boundary_weights[cluster_index][p]*(interval[p][1]-prototype[p][1])**2
            
        return distance
        
    def objective_function(self):
        value = 0

        for i in range(len(self.data)): # for each interval i 
            for k in range(len(self.prototype_vector)): # for each prototype k
                #if(len(data[i]) != len(prototype_vector[k])): 
                #    raise ValueError("data and prototype don't have the same dimensions")
                
                #distance component
                for p in range(len(self.data[i])): # for each feature
                    value += self.membership_matrix[i][k]*self.interval_distance(i, k)
                #entropy component
                if(self.membership_matrix[i][k] > 1e-10): value += self.T_u*self.membership_matrix[i][k]*gmpy2.log(self.membership_matrix[i][k]) #approx x*logx = 0 at x = 0
            
        return value
        
    def run(self, data, cluster_num, T_u=1, max_iterations=50, threshold=1e-10):       
        random.seed()

        # Initialization

        self.data = data
        self.cluster_num = cluster_num
        feature_num = len(data[0]) # Supposes the dataset is set up correctly
        data_num = len(data)
        self.prototype_vector = []
        self.lower_boundary_weights = [[1 for row in range(feature_num)] for col in range(cluster_num)]
        self.upper_boundary_weights = [[1 for row in range(feature_num)] for col in range(cluster_num)]
        self.membership_matrix = [[0 for col in range(cluster_num)] for row in range(data_num)]
        self.adequacy_history = []
        self.T_u = T_u

        t = 0

        starting_prototypes = random.sample(range(data_num), cluster_num)
        for i in starting_prototypes: 
            self.prototype_vector.append(data[i])
            
        starting_membership = random.choices(range(cluster_num), k=data_num)
        for i, k in enumerate(starting_membership):
            self.membership_matrix[i][k] = 1
            
        self.adequacy_history.append(self.objective_function()) # initial value of J
            
        while(t < max_iterations):
            
            t = t + 1
            
            #representation step
            for k in range(cluster_num):
                for j in range(feature_num):
                    #lower interval
                    weight_sum = 0
                    sum = 0
                    for i in range(data_num):
                        weight_sum += self.membership_matrix[i][k]
                    for i in range(data_num):
                        sum += self.membership_matrix[i][k]*(data[i][j][0]/weight_sum)
                        
                    self.prototype_vector[k][j][0] = sum/weight_sum
                    #upper interval
                    weight_sum = 0
                    sum = 0
                    for i in range(data_num):
                        weight_sum += self.membership_matrix[i][k]
                    for i in range(data_num):
                        sum += self.membership_matrix[i][k]*(data[i][j][1]/weight_sum)
                    
                    self.prototype_vector[k][j][1] = sum
                    
            #weighting step
            for k in range(cluster_num):
                lower_distance = []
                upper_distance = []
                for j in range(feature_num):
                    lower_value = 0
                    upper_value = 0
                    for i in range(data_num):
                        lower_value += self.membership_matrix[i][k]*(self.data[i][j][0]-self.prototype_vector[k][j][0])**2
                        upper_value += self.membership_matrix[i][k]*(self.data[i][j][1]-self.prototype_vector[k][j][1])**2
                    lower_distance.append(lower_value)
                    upper_distance.append(upper_value)
                
                for j in range(feature_num):
                    prod = 1
                    prod *= (lower_distance[j])**(1/(2*feature_num))
                    prod *= (upper_distance[j])**(1/(2*feature_num))
                    
                for j in range(feature_num):
                    # if division by zero is found, ignore until next step
                    if(lower_distance[j] > 1e-10): self.lower_boundary_weights[k][j] = prod/lower_distance[j]
                    if(upper_distance[j] > 1e-10): self.upper_boundary_weights[k][j] = prod/upper_distance[j]
                    
            #allocation step
            for i in range(data_num):
                denominator = 0
                for k in range(cluster_num):
                    denominator += gmpy2.exp(-self.interval_distance(i, k)/self.T_u)
                for k in range(cluster_num):
                    self.membership_matrix[i][k] = gmpy2.exp(-self.interval_distance(i, k)/self.T_u)/denominator
                    
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
            
    def crispy_membership(self): 
        '''
        Converts the fuzzy membership matrix U into a discrete one
        '''
        hard_membership_matrix = [[0 for col in range(self.cluster_num)] for row in range(len(self.data))]
        for i in range(len(self.membership_matrix)):
            k = np.argmax(self.membership_matrix[i])
            hard_membership_matrix[i][k] = 1
        return hard_membership_matrix
    
    def confusion_matrix(self, true_classes):
        crispy_m = self.crispy_membership()
        pred = [[n for n, m in enumerate(crispy_m[i]) if m==1][0] for i in range(len(self.data))]
        cm = confusion_matrix(true_classes, pred, labels=range(self.cluster_num))
        
        return cm
    
    def predict(self):
        crispy_m = self.crispy_membership()
        prediction = [[n for n, m in enumerate(crispy_m[i]) if m==1][0] for i in range(len(self.data))]
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
            data_point.append([gmpy2.mpfr(array[i]),gmpy2.mpfr(array[i+1])])
            
        structured_data.append(data_point)
        classes.append(int(array[0]))
    
    return structured_data, classes

def main():
    gmpy2.get_context().precision = 200
    
    data, classes = loaddotmat('Car.mat')
    model = ICMKmodel()
    model = model.run_many(data, max(classes), max_iterations=50, reinitializations=5, T_u = 0.1)
    
    model.analyze()
    
    print(f"RAND SCORE AJUSTADO: ", adjusted_rand_score(classes, model.predict()))
    
    cm = model.confusion_matrix(classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=range(model.cluster_num))
    disp.plot()
    plt.show() 
    
    #print(data)

if __name__ == "__main__":
    main()