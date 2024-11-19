#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>
#include "../include/csv.hpp"
using namespace std;

long long int CSVFile::get_row_num() {
    return row_number;
}

long long int CSVFile::get_column_num() {
    return column_number;
}

string CSVFile::get_value(long long int row, long long int column) {
    return values[row][column];
}

string CSVFile::get_name(long long int column) {
    return names[column];
}

vector<string> CSVFile::get_row(long long int row) {
    return values[row];
}

vector<string> CSVFile::get_column(long long int column) {
    vector<string> temp;
    for(long long int i = 0; i<get_row_num(); i++) {
        temp.push_back(values[i][column]);
    }
    return temp;
}

vector<vector<string>> CSVFile::get_columns(long long int columns) {
    vector<vector<string>> temp;
    for(long long int i = 0; i<get_row_num(); i++) {
        temp.push_back(vector<string>(values[i].begin(),values[i].begin()+columns));
    }
    return temp;
}

void CSVFile::edit_value(long long int row, long long int column, string newvalue) {
    if(row<get_row_num() && row >= 0 && column<get_column_num() && column >= 0) {
        values[row][column] = newvalue;
    }
    return;
}

void CSVFile::append_row(vector<string> row) {
    if(row.size() == column_number) {
        values.push_back(row);
    }
    row_number++;
    return;
}

void CSVFile::clear() {
    for(long long int i = 0; i<get_row_num(); i++) {
        values[i].clear();
    }
    values.clear();
    names.clear();
    row_number = 0;
    column_number = 0;
    last_read.clear();
}

//Limpa o CSVFile atual e lê as informações localizadas em filename.csv.
//Essa implementação não funciona com valores que são strings que possuem vírgula e ignora aspas.
void CSVFile::read_file(string filename) {

    clear();

    fstream fin; 

    fin.open(filename+".csv", ios::in);
    if(fin.fail()) return;
    
    // Definindo os nomes das colunas do CSV (primeira linha)

    vector<string> row; 
    string line, words;

    getline(fin, line);
    stringstream columns(line);

    while(getline(columns, words, ',')) { 
        row.push_back(words); 
    }

    if(column_number == 0) column_number = row.size();

    names = row;

    // Obtendo os valores
  
    while (getline(fin, line)) {

        row.clear(); 
        stringstream columns(line);
  
        while(getline(columns, words, ',')) { 
            row.push_back(words); 
        }

        append_row(row);
    }

    last_read = filename;

}

//Limpa o CSVFile atual e lê um número rows de linhas localizadas em filename.csv.
//Essa implementação não funciona com valores que são strings que possuem vírgula e ignora aspas.
void CSVFile::read_file(string filename, long long int rows) {

    clear();

    fstream fin; 

    fin.open(filename+".csv", ios::in);
    if(fin.fail()) return;
  
    vector<string> row; 
    string line, words;

    // Definindo os nomes das colunas do CSV (primeira linha)

    getline(fin, line);
    stringstream columns(line);

    while(getline(columns, words, ',')) { 
        row.push_back(words); 
    }

    if(column_number == 0) column_number = row.size();

    names = row;

    // Obtendo os valores até a "rows"-ésima linha
  
    for(long long int i = 0; i<rows; i++) {

        getline(fin, line);

        row.clear(); 
        stringstream columns(line);
  
        while(getline(columns, words, ',')) { 
            row.push_back(words); 
        }

        append_row(row);
        row_number++;
    }

    last_read = filename;

}

CSVFile::CSVFile(string filename) {
    row_number = 0;
    column_number = 0;
    read_file(filename);
}

CSVFile::CSVFile(long long int columns) {
    row_number = 0;
    column_number = columns;
}

CSVFile::CSVFile() {
    row_number = 0;
    column_number = 0;
}

// Gera um arquivo filename.csv no diretório atual usando as informações do CSVFile selecionado
void CSVFile::create_dotcsv(string filename) { 

	fstream fout; 

	fout.open(filename+".csv", ios::out | ios::app); 

	for(long long int i = 0; i<row_number; i++) {
        for(long long int j = 0; j<column_number-1; j++) {
            fout << values[i][j] << ',';
        }
        fout << values[i][column_number-1] << '\n';
	} 
}

// Substitui as informações em filename.csv com as do CSVFile selecionado
void CSVFile::substitute_dotcsv(string filename) {
    create_dotcsv(filename+"_new");
    remove((filename+".csv").c_str());
    rename((filename+"_new.csv").c_str(), (filename+".csv").c_str());
}

void CSVFile::print_file() {
    for(long long int i = 0; i<row_number; i++) {
        for(long long int j = 0; j<column_number-1; j++) {
            cout << values[i][j] << ',';
        }
        cout << values[i][column_number-1] << '\n';
	} 
}
