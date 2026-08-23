// Device-side rank-two Lorentz tensor. The class intentionally contains only
// low-level algebra; complete process Waves belong under process/waves/.
#ifndef CTPWA_FRAMEWORK_TENSORS_TENSOR_CUH
#define CTPWA_FRAMEWORK_TENSORS_TENSOR_CUH

#include "framework/math/FourVector.cuh"
#include "framework/math/Lorentz.cuh"

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

        // A^{mu nu} v_nu: the second tensor index is contracted with the
        // covariant vector. TensorContraction.cuh provides the equivalent
        // named helper for code where the index choice should be explicit.
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

        __device__ static tensor Gnormal(){
            tensor temp1;
            temp1._matrix[0][0] = 1.0; temp1._matrix[1][1] = -1.0; temp1._matrix[2][2] = -1.0; temp1._matrix[3][3] = -1.0;
            return temp1;
        }

        __device__ static double epsilon(int mu, int nu, int rho, int sigma){
            return ctpwa::levi_civita(mu, nu, rho, sigma);
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
#endif // CTPWA_FRAMEWORK_TENSORS_TENSOR_CUH
