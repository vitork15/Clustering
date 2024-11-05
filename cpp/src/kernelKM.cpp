#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include "../include/csv.hpp"
using namespace std;

int main() {
    // loading dataset using CSV
    CSVFile iris_dataset = CSVFile("iris");

    vector<string> types = iris_dataset.get_column(4);
    for(int i = 0; i<types.size(); i++) {
        cout << types[i] << endl;
    }

    return 0;
}