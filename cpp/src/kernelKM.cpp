#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include "../include/csv.hpp"
using namespace std;

int main() {

    CSVFile iris_dataset = CSVFile("iris");

    //iris_dataset.print_file();
    cout << iris_dataset.get_row_num() << endl;
    cout << iris_dataset.get_column_num() << endl;


}