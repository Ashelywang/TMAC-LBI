//
//  splitting.h
//  draft
//
//  Created by blues cookie on 16/6/30.
//  Copyright © 2016年 blues cookie. All rights reserved.
//

#ifndef splitting_h
#define splitting_h

#include"splitting_schemes.h"
 
//Backward_kappa_Forward_Splitting
// z^{k+1} = z^k - eta_k gradient( kappa * prox(z^k) )
// which can be simplified to the following
// z^{k+1} = z^k - eta_k gradient( kappa * prox(z^k) )


template <typename Backward, typename Gradient>
class Backward_kappa_Forward_Splitting: public SchemeInterface {
public:
    Backward backward;
    Gradient gradient;
    Vector *x;
    Vector theta; // each new operator will have a copy of this guy
    Vector y;
    double kappa ;
    double relaxation_step_size;
    
    Backward_kappa_Forward_Splitting<Backward, Gradient>(Vector* x_, Backward backward_, Gradient gradient_, double  relaxation_step_size_, double kappa_ = 200) {
        x = x_;
        backward = backward_;
        gradient = gradient_;
        relaxation_step_size = relaxation_step_size_;
        theta.resize(x->size());
        kappa = kappa_;
        y.resize(x->size());
    }
    
    void update_params(Params* params) {
        // TODO: forward and backward might use different step sizes
        backward.update_step_size(params->get_step_size());
        //gradient.update_step_size(params->get_step_size());
        relaxation_step_size = params->get_tmac_step_size();
    }
    
    double operator() (int index) {
        // Step 1: get the old x[index]
        double old_x_at_idx = (*x)[index];
        // Step 2: apply the backward operator on x and save it to theta
        
        // ###Create a variable, which is a copy of theta.
        Vector tmp = theta;
        
        // ###tmp  = Prox(x)
        backward(x, &tmp);
        /*if((*x)[0]>1 && index ==0){
            cout <<"X= " <<flush;
            for(int i =0;i<20;i++){
                cout<<(*x)[i]<<" "<<flush;
            }
            
            cout<<endl<<"tmp[0] = "<<tmp[0]<<endl;
        }*/
        // ###tmp = tmp * kappa
        scale(tmp, kappa);
        
        /*if((*x)[0]>1&& index ==0){
            cout<<"tmp[0]*kappa = "<<tmp[0]<<endl;
        }*/
        //cout << " The backward(x, &theta) is "<<tmp[index]<<endl;
       
        
        //Vector tmpvec(theta);//Create a tmp to store theta and thera*kappa
        
        
        // Step 3: then apply the gradient operator on theta
        double forward_grad_at_theta = gradient(&tmp, index);
        //cout<<"the grad is  "<<forward_grad_at_theta<<endl;
        // Step 4: Compute z using z^{k+1} = z^k - eta_k y_k
        relaxation_step_size = 2.7606e-05;
        //double S_i = old_x_at_idx - relaxation_step_size*forward_grad_at_theta;
        double S_i = forward_grad_at_theta;
        //cout<<"The relaxation step size is "<<relaxation_step_size<<endl<<endl;
        
        
                //cout<< "scale(theta, kappa) =  "<<theta[index]<<endl<<endl;
        // Step 5: get the most recent x[index]
        old_x_at_idx = (*x)[index];
        // Step 4: update z[index]
        (*x)[index] -= relaxation_step_size*S_i;
        /*if((*x)[0]>1&& index ==0){
            cout<<"relaxation_step_size*S_i = "<<relaxation_step_size*S_i<<endl;
            cout <<"(*x)[index] - relation*Si = " <<(*x)[index]<<endl;
        }*/
        //Update theta
        backward(x, &theta);
        /*if((*x)[0]>1&& index ==0){
            cout <<"backward prox(x) = " <<theta[0]<<endl;
        }*/
        scale(theta, kappa);
        /*if((*x)[0]>1&& index ==0){
            cout <<"kappa * backward prox(x) = " <<theta[0]<<endl<<endl;
        }*/

        return S_i;
    }
    
    // TODO: implement this for sync-operator
    void operator()(int index, double &S_i) {
    }
    
    void update(Vector& s, int range_start, int num_cords) {
        for (size_t i = 0; i < num_cords; ++i ) {
            (*x)[i+range_start] -= relaxation_step_size * s[i];
        }
    }
    
    void update (double s, int idz ) {
        (*x)[idz] -= relaxation_step_size * s;
    }
    
    void update_cache_vars (int rank, int index ) {
    }
    
};








#endif /* splitting_h */
