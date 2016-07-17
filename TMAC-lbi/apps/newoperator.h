//
//  newoperator.h
//  draft
//
//  Created by blues cookie on 16/6/30.
//  Copyright © 2016年 blues cookie. All rights reserved.
//

#ifndef newoperator_h
#define newoperator_h
#include "operators.h"



template<typename Mat>
class grad_for_square_loss: public OperatorInterface {
public:
    Mat* A;
    Vector *b, *Atx;
    Mat* At; // this is for the sync-parallel stuff
    double weight;
    double step_size;
    double n;
    
    
    
    double operator() (Vector* x, int index) {
        // calculate the forward step
        trans_multiply(*A, *x, *Atx);
        double A_iAtx = dot(A, Atx, index);
        
        double A_ib = dot(A, b, index);
        double grad_for_sq_at_i = 2.0 * (A_iAtx - A_ib)/n;
        return grad_for_sq_at_i;
    }
    
    void operator() (Vector* v_in, Vector* v_out) {
        int m = A->rows(), n = A->cols();
        Vector Atv_in(m, 0.);
        trans_multiply(*A, *v_in, Atv_in);
        add(Atv_in, *b, -1.);
        Vector temp(n, 0.);
        multiply(*A, Atv_in, temp);
        scale(temp, 2.0/n);
        for (int i = 0; i < m; i++) {
            (*v_out)[i] = temp[i];
        }
    }
    
    double operator() (double val, int index = 1) {
        return DOUBLE_MARKER;
    }
    
    void update_step_size(double step_size_) {
        step_size = step_size_;
    }
    
    void update_cache_vars(double old_x_i, double new_x_i, int index) {
        add(Atx, A, index, -old_x_i + new_x_i);
    }
    
    void update_cache_vars(Vector* x, int rank, int num_threads){
        int m = At->rows(); //y=A'*x
        int block_size = m/num_threads;
        int start_idx = rank*(block_size);
        int end_idx = (rank == num_threads-1)? m : start_idx+block_size;
        for(int iter=start_idx; iter != end_idx; ++iter){
            (*Atx)[iter]=dot(At, x, iter);
        }
    }
    
    grad_for_square_loss () {
        step_size = 1.;
        weight = 1.;
    }
    grad_for_square_loss (double step_size_, double weight_ = 1.) {
        step_size = step_size_;
        weight = weight_;
    }
    
    grad_for_square_loss(Mat* A_, Vector* b_, Vector* Atx_,
                                  double step_size_ = 1., double weight_ = 1.,
                                  Mat* At_ = nullptr) {
        step_size = step_size_;
        weight = weight_;
        A = A_;
        b = b_;
        Atx = Atx_;
        At = At_;
        n = A->cols();
    }
};




#endif /* newoperator_h */
