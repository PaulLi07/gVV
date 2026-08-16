#ifndef TENSOR_H
#define TENSOR_H

#include "FV.h"

class tensor{
    public:
        double _matrix[4][4];


        __device__ tensor(FV fv1, FV fv2){
            for(int i=0;i<4;i++){
               for(int j=0;j<4;j++){
                    _matrix[i][j] = fv1.Get(i)*fv2.Get(j);
                } 
            }
        }

        __device__ tensor(){
            for(int i=0;i<4;i++){
               for(int j=0;j<4;j++){
                    _matrix[i][j] = 0.0;
                } 
            }
        }

        __device__ tensor(const tensor& obj){
            for(int i=0;i<4;i++){
               for(int j=0;j<4;j++){
                    _matrix[i][j] = obj._matrix[i][j];
                }
            }
        }

        __device__ tensor& operator=(const tensor& other){
            if(this == &other){
                return *this;
            }
            for(int i=0;i<4;i++){
               for(int j=0;j<4;j++){
                    _matrix[i][j] = other._matrix[i][j];
                }
            }
            return *this;
        }

        __device__ tensor operator+(tensor obj) const {
            tensor temp;
            for(int i=0;i<4;i++){
               for(int j=0;j<4;j++){
                    temp._matrix[i][j] = _matrix[i][j] + obj._matrix[i][j];
                }
            }
            return temp;
        }

        __device__ tensor operator-(tensor obj) const {
            tensor temp;
            for(int i=0;i<4;i++){
               for(int j=0;j<4;j++){
                    temp._matrix[i][j] = _matrix[i][j] - obj._matrix[i][j];
                }
            }
            return temp;
        }

        __device__ tensor operator*(double scale) const {
            tensor temp;
            for(int i=0;i<4;i++){
               for(int j=0;j<4;j++){
                    temp._matrix[i][j] = (scale)*(_matrix[i][j]);
                }
            }
            return temp;
        }

        __device__ tensor operator/(double scale) const {
            tensor temp;
            for(int i=0;i<4;i++){
               for(int j=0;j<4;j++){
                    temp._matrix[i][j] = _matrix[i][j]/scale;
                }
            }
            return temp;
        }

        //NOTICE! Defaultly, this operator choose the second index of tensor!
        //It does not matter when contract with symmetric tensor, like Projection
        __device__ FV  operator*(FV obj) const {
            FV temp(0,0,0,0);

            for(int i=0;i<4;i++){
               for(int j=0;j<4;j++){
                    if(j==0){temp.Four_momentum[i] = temp.Four_momentum[i] + _matrix[i][j]*obj.Get(j);}
                    else{temp.Four_momentum[i] = temp.Four_momentum[i] - _matrix[i][j]*obj.Get(j);}
                }
            }

            return temp;
        }

        __device__ static tensor Gwavetide(FV fv){
            tensor temp1;
            temp1._matrix[0][0] = 1.0; temp1._matrix[1][1] = -1.0; temp1._matrix[2][2] = -1.0; temp1._matrix[3][3] = -1.0;
            tensor temp2(fv,fv);
            temp2 = temp2/(fv*fv);
            return (temp1-temp2);
        }

        __device__ static FV Projection_Pwave(FV mom, FV r){
            tensor gwavetide = Gwavetide(mom);
            FV t = gwavetide*r;
            return t;
        }

        __device__ static tensor Projection_Dwave(FV mom, FV r){
            tensor gwavetide = Gwavetide(mom);
            tensor part1 = tensor(gwavetide*r,gwavetide*r);
            tensor part2 = gwavetide*((gwavetide*r)*r);
            tensor t = part1-part2/3.0;
            return t;
        }

        __device__ static tensor Gnormal(){
            tensor temp1;
            temp1._matrix[0][0] = 1.0; temp1._matrix[1][1] = -1.0; temp1._matrix[2][2] = -1.0; temp1._matrix[3][3] = -1.0;
            return temp1;
        }

        __device__ static double epsilon(int mu, int nu, int rho, int sigma){
            if(mu==0&&nu==1&&rho==2&&sigma==3){return +1.;}
            if(mu==0&&nu==1&&rho==3&&sigma==2){return -1.;}
            if(mu==0&&nu==2&&rho==1&&sigma==3){return -1.;}
            if(mu==0&&nu==2&&rho==3&&sigma==1){return +1.;}
            if(mu==0&&nu==3&&rho==1&&sigma==2){return +1.;}
            if(mu==0&&nu==3&&rho==2&&sigma==1){return -1.;}
            if(mu==1&&nu==0&&rho==2&&sigma==3){return -1.;}
            if(mu==1&&nu==0&&rho==3&&sigma==2){return +1.;}
            if(mu==1&&nu==2&&rho==0&&sigma==3){return +1.;}
            if(mu==1&&nu==2&&rho==3&&sigma==0){return -1.;}
            if(mu==1&&nu==3&&rho==0&&sigma==2){return -1.;}
            if(mu==1&&nu==3&&rho==2&&sigma==0){return +1.;}
            if(mu==2&&nu==0&&rho==1&&sigma==3){return +1.;}
            if(mu==2&&nu==0&&rho==3&&sigma==1){return -1.;}
            if(mu==2&&nu==1&&rho==0&&sigma==3){return -1.;}
            if(mu==2&&nu==1&&rho==3&&sigma==0){return +1.;}
            if(mu==2&&nu==3&&rho==0&&sigma==1){return +1.;}
            if(mu==2&&nu==3&&rho==1&&sigma==0){return -1.;}
            if(mu==3&&nu==0&&rho==1&&sigma==2){return -1.;}
            if(mu==3&&nu==0&&rho==2&&sigma==1){return +1.;}
            if(mu==3&&nu==1&&rho==0&&sigma==2){return +1.;}
            if(mu==3&&nu==1&&rho==2&&sigma==0){return -1.;}
            if(mu==3&&nu==2&&rho==0&&sigma==1){return -1.;}
            if(mu==3&&nu==2&&rho==1&&sigma==0){return +1.;}
            return 0.;
        }

        //Contract with the later two index!
        __device__ static tensor Epsilon(FV p, FV q){
            tensor temp1;

            for(int i=0;i<4;i++){
                for(int j=0;j<4;j++){
                    for(int k=0;k<4;k++){
                        for(int l=0;l<4;l++){
                            int sign = 1;
                            if(k==0 && l==0){sign = +1;}
                            if(k!=0 && l==0){sign = -1;}
                            if(k==0 && l!=0){sign = -1;}
                            if(k!=0 && l!=0){sign = +1;}

                            temp1._matrix[i][j] = temp1._matrix[i][j] + sign*epsilon(i,j,k,l)*p.Get(k)*q.Get(l);
                        }
                    }
                }
            }

            return temp1;

        }

        __device__ void print(){
            for(int i=0;i<4;i++){
               for(int j=0;j<4;j++){
                    printf("%d %d:  %f",i,j,_matrix[i][j]);
                }
            }
        }

};
#endif // TENSOR_H
