#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>
#include <set>
#include <map>
#include <iomanip>
#include "../include/csv.hpp"
using namespace std;

double euclidian_distance(vector<double>& value1, vector<double>& value2) {
    double distance = 0;
    for(long long int i = 0; i<value1.size(); i++) {
        distance += (value1[i]-value2[i])*(value1[i]-value2[i]);
    }
    return distance;
}

double gaussian_kernel_local(vector<double>& value, vector<double>& prototype, long long int cluster, vector<vector<double>>& width_local) {
    double distance = 0;
    for(long long int j = 0; j<value.size(); j++) {
        distance += ((value[j]-prototype[j])*(value[j]-prototype[j]))*width_local[cluster][j]/2;
    }
    return exp(-distance);
}

vector<vector<double>> string_to_double(vector<vector<string>>& v) {
    vector<vector<double>> result;
    result.resize(v.size());
    for(long long int i = 0; i<v.size(); i++) {
        result[i].resize(v[i].size());
        for(long long int j = 0; j<v[i].size(); j++) {
            result[i][j] = stod(v[i][j]);
        }
    }
    return result;
}

vector<long long int> classes_to_number(vector<string> classes) {
    map<string,long long int> words;
    vector<long long int> result;
    long long int c = 0;
    for(string i : classes) {
        if(words.find(i) == words.end()) words.insert(pair<string,long long int>(i,c++));
    }
    for(string i : classes) {
        result.push_back(words.at(i));
    }
    return result;
}

template <typename T>
long long int argmax(vector<T>& vec) {
    long long int result = 0;
    for(long long int i = 0; i<vec.size(); i++) {
        if(vec[i] > vec[result]) result = i;
    }
    return result;
}

template <typename T>
T max(vector<T> vec) {
    T res = vec[0];
    for(T e : vec) {
        if(e > res) res = e;
    }
    return res;
}

vector<vector<long long int>> confusion_matrix(vector<long long int> pred, vector<long long int> true_pred) {

    vector<vector<long long int>> cm;

    long long int size = max(pred) + 1;

    cm.resize(size);
    for(long long int i = 0; i<size; i++) cm[i].resize(size, 0);

    for(long long int i = 0; i<pred.size(); i++) {
        cm[true_pred[i]][pred[i]]++;
    }

    return cm;
}

class kernelKM_PL {
    private:
        vector<vector<double>> prototype_vector;
        vector<vector<double>> membership_vector;
        vector<vector<double>> width_matrix;
        vector<vector<double>> dataset;
        vector<double> adequacy_history;
        long long int cluster_num;
        long long int feature_num;
        long long int data_num;
        double fuzzifier;

    public:
        kernelKM_PL() {
            cluster_num = 1;
            fuzzifier = 1.0;
        }
        void debug_adequacy() {
            double v = adequacy_history[0];
            cout << setprecision(15) << "J INICIAL: " << v << endl;
            for(long long int i = 1; i<adequacy_history.size(); i++) {
                if(adequacy_history[i] > adequacy_history[i-1]) cout << setprecision(15) << "J AUMENTOU NA ITERACAO " << i << ": " << adequacy_history[i-1] << " -> " << adequacy_history[i] << endl;
                else cout << setprecision(15) << "J REDUZIU NA ITERACAO " << i << ": " << adequacy_history[i-1] << " -> " << adequacy_history[i] << endl;
            }
        }
        vector<vector<double>> return_fuzzy_membership() {
            return membership_vector;
        }
        vector<vector<double>> return_membership() {
            vector<vector<double>> result;
            result.resize(data_num);
            for(long long int i = 0; i<data_num; i++) {
                result[i].resize(cluster_num, 0.0);
                result[i][argmax<double>(membership_vector[i])] = 1.0;
            }
            return result;
        }
        vector<long long int> return_pred() {
            vector<long long int> result;
            result.resize(data_num);
            for(long long int i = 0; i<data_num; i++) {
                result[i] = argmax<double>(membership_vector[i]);
            }
            return result;
        }
        double return_final_adequacy() {
            return adequacy_history.back();
        }
        double calculate_adequacy() {
            /*data_num = len(self.dataset)
            adequacy = 0
            for k in range(self.cluster_num):
                for i in range(data_num):
                    adequacy += (self.membership_vector[i][k]**self.fuzzifier)*2*(1-self.kernel(self.dataset[i],self.prototype_vector[k],k,self.width_matrix))
            return adequacy*/
            double adequacy = 0;
            for(long long int k = 0; k<cluster_num; k++) {
                for(long long int i = 0; i<data_num; i++) {
                    adequacy += 2*pow(membership_vector[i][k], fuzzifier)*(1-gaussian_kernel_local(dataset[i], prototype_vector[k], k, width_matrix));
                }
            }
            return adequacy;
        }
        void fit(vector<vector<double>> data, long long int clusters, double fuzzifier, long long int max_iterations, double epsilon, double gamma, double theta) {


            // inicializando variáveis

            adequacy_history.clear();
            prototype_vector.clear();

            dataset = data;
            cluster_num = clusters;
            feature_num = dataset[0].size();
            data_num = dataset.size();
            width_matrix.resize(cluster_num);
            for(long long int k = 0; k<cluster_num; k++) {
                width_matrix[k].resize(feature_num, 1.0);
            }
            membership_vector.resize(data_num);
            for(long long int i = 0; i<data_num; i++) {
                membership_vector[i].resize(cluster_num, 0.0);
            }

            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<long long int> proto(0,data_num-1);
            uniform_int_distribution<long long int> clust(0,cluster_num-1);

            set<long long int> initial_prototypes;
            // selecionando numeros unicos aleatórios para os prototipos
            while(initial_prototypes.size() < cluster_num) {
                initial_prototypes.insert(proto(gen));
            }
            for(long long int i : initial_prototypes) {
                prototype_vector.push_back(dataset[i]);
            }
            // selecionando partições aleatorias para os pontos
            for(long long int i = 0; i<data_num; i++) {
                membership_vector[i][clust(gen)] = 1.0;
            }

            adequacy_history.push_back(calculate_adequacy());

            long long int t = 0;
            vector<double> theta_values, distance_to_k;
            vector<long long int> above_theta, below_theta, zeros;
            double sum, total_sum, prod, correction_factor;

            while(true) {
                t = t + 1;

                // Step 1: Adjusting width matrix
                for(long long int k = 0; k<cluster_num; k++) {
                    theta_values.clear();
                    theta_values.resize(feature_num);
                    above_theta.clear();
                    below_theta.clear();

                    for(long long int j = 0; j<feature_num; j++) {
                        sum = 0;
                        for(long long int i = 0; i<data_num; i++) {
                            sum += pow(membership_vector[i][k], fuzzifier)*gaussian_kernel_local(dataset[i], prototype_vector[k], k, width_matrix)*(dataset[i][j]-prototype_vector[k][j])*(dataset[i][j]-prototype_vector[k][j]);
                        }
                        theta_values[j] = sum;
                        if(sum < theta) below_theta.push_back(j);
                        else above_theta.push_back(j);
                    }

                    if(below_theta.size() == 0) {
                        prod = 1;
                        for(long long int j = 0; j<feature_num; j++) {
                            prod *= theta_values[j];
                        }
                        prod = pow(gamma*prod, 1.0/feature_num);
                        for(long long int j = 0; j<feature_num; j++) {
                            width_matrix[k][j] = prod/theta_values[j];
                        }
                    }

                    else {
                        correction_factor = 1;
                        for(long long int j : below_theta) {
                            correction_factor *= 1/width_matrix[k][j];
                        }
                        prod = 1;
                        for(long long int j : above_theta) {
                            prod *=  theta_values[j];
                        }
                        prod = pow(gamma*correction_factor*prod, 1.0/(feature_num-below_theta.size()));
                        for(long long int j : above_theta) {
                            width_matrix[k][j] = prod/theta_values[j];
                        }
                    }
                }
                // Step 2: Using new width matrix to calculate new best prototypes
                for(long long int k = 0; k<cluster_num; k++) {
                    for(long long int j = 0; j<feature_num; j++) {
                        sum = 0;
                        for(long long int i = 0; i<data_num; i++) {
                            sum += pow(membership_vector[i][k], fuzzifier)*gaussian_kernel_local(dataset[i], prototype_vector[k], k, width_matrix);
                        }
                        if(sum > theta) {
                            total_sum = 0;
                            for(long long int i = 0; i<data_num; i++) {
                                total_sum += pow(membership_vector[i][k], fuzzifier)*gaussian_kernel_local(dataset[i], prototype_vector[k], k, width_matrix)*dataset[i][j];
                            }
                            prototype_vector[k][j] = total_sum/sum;
                        }
                    }
                }
                // Step 3: Using new prototypes to calculate best fuzzy partitions
                for(long long int i = 0; i<data_num; i++) {

                    distance_to_k.clear();
                    distance_to_k.resize(cluster_num);
                    zeros.clear();

                    for(long long int k = 0; k<cluster_num; k++) {
                        distance_to_k[k] = 1 - gaussian_kernel_local(dataset[i], prototype_vector[k], k, width_matrix);
                        if(distance_to_k[k] == 0) zeros.push_back(k);
                    }

                    if(zeros.size() == 0) {
                        for(long long int k = 0; k<cluster_num; k++) {
                            sum = 0;
                            for(long long int h = 0; h<cluster_num; h++) {
                                sum += pow((distance_to_k[k]/distance_to_k[h]), 1.0/(fuzzifier-1));
                            }
                            membership_vector[i][k] = 1.0/sum;
                        }
                    }
                    else {
                        for(long long int k = 0; k<cluster_num; k++) {
                            membership_vector[i][k] = 0;
                        }
                        for(long long int k : zeros) {
                            membership_vector[i][k] = 1.0/zeros.size();
                        }
                    }
                }

                adequacy_history.push_back(calculate_adequacy());


                // Step 4: Check stopping condition
                if(t > max_iterations || abs((adequacy_history[adequacy_history.size()-1] - adequacy_history[adequacy_history.size()-2]))<epsilon) break;
            }
            return;
        }
        void fit_reinit(vector<vector<double>> data, long long int clusters, double fuzzifier, long long int max_iterations, double epsilon, double gamma, double theta, long long int reinitializations) {
            fit(data, clusters, fuzzifier, max_iterations, epsilon, gamma, theta);
            vector<vector<double>> save_prototype_vector = prototype_vector;
            vector<vector<double>> save_membership_vector = membership_vector;
            vector<vector<double>> save_width_matrix = width_matrix;
            vector<double> save_adequacy_history = adequacy_history;
            double criterion = return_final_adequacy();
            long long int reinit = 0;
            while(reinit++ < reinitializations) {
                fit(data, clusters, fuzzifier, max_iterations, epsilon, gamma, theta);
                if(return_final_adequacy() < criterion) {
                    save_prototype_vector = prototype_vector;
                    save_membership_vector = membership_vector;
                    save_width_matrix = width_matrix;
                    save_adequacy_history = adequacy_history;
                    criterion = return_final_adequacy();
                }
            }
            prototype_vector = save_prototype_vector;
            membership_vector = save_membership_vector;
            width_matrix = save_width_matrix;
            adequacy_history = save_adequacy_history;
            return;
        }
};

int main() {
    // loading dataset using CSV
    CSVFile iris_dataset = CSVFile("iris");

    vector<vector<string>> data = iris_dataset.get_columns(4);
    vector<string> true_types = iris_dataset.get_column(4);
    
    vector<vector<double>> dataf = string_to_double(data);


    kernelKM_PL model;
    //model.fit(dataf, 3, 1.05, 500, 1e-12, 1, 1e-8);
    model.fit_reinit(dataf, 3, 1.1, 300, 1e-12, 1, 1e-10, 500);

    vector<long long int> result = model.return_pred();
    vector<long long int> numbers = classes_to_number(true_types);

    vector<vector<long long int>> cm = confusion_matrix(result, numbers);
    for(long long int i = 0; i<cm.size(); i++) {
        for(long long int j = 0; j<cm[i].size(); j++) {
            cout << cm[i][j] << " ";
        }
        cout << endl;
    }

    cout << "J = " << model.return_final_adequacy() << endl;

    model.debug_adequacy();

    /*for(long long int i = 0; i<result.size(); i++) {
        for(long long int j = 0; j<result[i].size(); j++) {
            cout << result[i][j] << " ";
        }
        cout << endl;
    }*/

    return 0;
}