//
//  main.cpp
//  draft
//
//  Created by blues cookie on 16/6/29.
//  Copyright © 2016年 blues cookie. All rights reserved.
//

#define _theta
#include <iostream>
#include"newoperator.h"
#include"splitting.h"
#include "matrices.h"
#include "algebra.h"
#include "operators.h"
#include "parameters.h"
#include "splitting_schemes.h"
#include "tmac.h"
#include "util.h"
#include "MarketIO.h"
#include "../lib/Eigen/Dense"
#include <thread>
#include<complex>
using namespace std;
#include "algebra_namespace_switcher.h"
/*
template<typename T>
void Controller<T>::my_record_path(std::thread::id id, Vector& path, int itr) {
    auto& w = worker_state.at(id);
    Vector* tmp = &w.scheme->theta;
    int len = tmp->size();
    path.insert(path.begin() + itr * len, tmp->begin(), tmp->end());
}
*/
double objective(Vector& Atx, Vector& b);

/***********************************************************
 LBI:
 for the problem
        y = A' * \theta + \epsilon
 Use LBI to find a iteration path
 ***********************************************************/

double my_norm(Matrix& Mat){
    int row=Mat.rows();
    int col=Mat.cols();
    Eigen::MatrixXd A(row,col);
    for(int i=0;i<row;++i)
        for(int j=0;j<col;++j)
            A(i,j)=Mat(i,j);
    
    return A.squaredNorm();
}


int main(int argc, char * argv[]) {
    // Steo 0: Define the parameters and input file names
    Params params;
    double kappa = 200.0;
    
    string data_file_name;
    string label_file_name;
    
    // Step 1. Parse the input argument
    parse_input_argv_mm(&params, argc, argv, data_file_name, label_file_name,kappa);
    
    // Step 2. Load the data or generate synthetic data, define matained variables
    Matrix A;   // matrix is a row major matrix
    Vector y;
    
    loadMarket(A, data_file_name);
    loadMarket(y, label_file_name);
    
    // set para
    int problem_size = A.rows();
    params.problem_size = problem_size;
    params.tmac_step_size = 0.5;
    //Added!!!
    params.use_controller = true; //added to record path
    params.worker_type = "random";
    int sample_size = A.cols();
    // define auxilary variables
    Vector x(problem_size, 0.);   // unknown variables, initialized to zero
    Vector Atx(sample_size, 0.);  // maintained variables, initialized to zero
    
    
    // Step 3. Define your forward, or backward operators based on data and parameters
    double normAtA = 181.11; //to be modified
    cout<<"The norm of AtA is    "<<normAtA<<endl<<endl;
    double operator_step_size = 1.0/kappa/normAtA;
    cout << "The operator step size is : "<<operator_step_size<<endl<<endl;
    params.step_size = 1.0;
 
    // backward operator
    prox_l1 backward(1.0);
    using Backward= decltype( backward );

    //grad operator
    
    grad_for_square_loss<Matrix>  grad(&A,&y,&Atx);
    using Gradient = decltype(grad);
    
    // Step 4. Define your operator splitting scheme
    
    Backward_kappa_Forward_Splitting<Backward,Gradient> bkfs(&x,  backward,grad, operator_step_size,kappa);
    
    // Step 6. Call the TMAC function
    
    double start_time = get_wall_time();
    TMAC(bkfs, params);
    double end_time = get_wall_time();
    
    print_parameters(params);
    
    cout << "For LBI the Computing time is: " << end_time - start_time << endl;
    // Step 7. Print results
    
    cout << "Objective value is: " << objective(Atx, y) << endl;
    cout << "---------------------------------" << endl;
    //Added!!!
    
    cout<<"Path :"<<endl;
    for(int i =0;i<A.rows();i++){
        cout<< path[i]<<" ";
    }

    cout<<endl<<endl<<endl;
    
    for(int i =0;i<A.rows();i++){
        cout<< path[i+A.rows()*14000]<<" ";
    }
    cout<<endl<<endl<<endl;
    cout<<path.size()<<endl<<endl;
    
    cout<<"The column of A is "<<A.cols()<<endl;
     cout<<"The rows of A is "<<A.rows()<<endl;
    //cout<<my_eigenvalue(A)<<endl;
    //print(path);//print recorded path
    return 0;

}

double objective(Vector& Atx, Vector& b) {
    
    int len = b.size();
    Vector temp = Atx;
    add(temp, b, -1.);
    double norm_of_temp = norm(temp);
    return 0.5 * norm_of_temp * norm_of_temp;
    
}

