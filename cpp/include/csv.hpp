#ifndef CSV_H
#define CSV_H
#include <string>
#include <vector>
#include <fstream>
using namespace std;

class CSVFile {
    private:
        vector<vector<string>> values;
        vector<string> names;
        long long int row_number, column_number;
        string last_read;
    public:
        long long int get_row_num();
        long long int get_column_num();
        string get_name(long long int column);
        string get_value(long long int row, long long int column);
        vector<string> get_row(long long int row);
        vector<string> get_column(long long int column);
        void edit_value(long long int row, long long int column, string newvalue);
        void append_row(vector<string> row);
        CSVFile(string filename);
        CSVFile(long long int columns);
        CSVFile();
        void clear();
        void read_file(string filename);
        void read_file(string filename, long long int rows);
        void create_dotcsv(string filename);
        void substitute_dotcsv(string filename);
        void print_file();
};

#endif