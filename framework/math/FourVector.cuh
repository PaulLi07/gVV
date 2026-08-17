// Device-side Lorentz four-vector using the (+---) metric and component order
// [E, px, py, pz]. ROOT input conversion into this order happens only in the
// process TermEvaluator.
#ifndef CTPWA_FRAMEWORK_MATH_FOUR_VECTOR_CUH
#define CTPWA_FRAMEWORK_MATH_FOUR_VECTOR_CUH

#include "framework/math/DeviceComplex.cuh"
#include <cstdio>

class FV
{
    public:

        double Four_momentum[4];

        __device__ FV(double E, double px, double py, double pz){
            Four_momentum[0] = E;
            Four_momentum[1] = px;
            Four_momentum[2] = py;
            Four_momentum[3] = pz;
        }

        __device__ FV(){
            Four_momentum[0] = 0;
            Four_momentum[1] = 0;
            Four_momentum[2] = 0;
            Four_momentum[3] = 0;
        }

        __device__ FV(const FV& obj){
            Four_momentum[0] = obj.Four_momentum[0];
            Four_momentum[1] = obj.Four_momentum[1];
            Four_momentum[2] = obj.Four_momentum[2];
            Four_momentum[3] = obj.Four_momentum[3];
        }

        __device__ void print(){
            printf("%f,%f,%f,%f\n", Four_momentum[0],Four_momentum[1],Four_momentum[2],Four_momentum[3]);
        }

        __device__ double Get(int idx) const {
            return Four_momentum[idx];
        }

        __device__ FV operator+(FV obj) const {
            FV temp(Four_momentum[0] + obj.Four_momentum[0], Four_momentum[1] + obj.Four_momentum[1], Four_momentum[2] + obj.Four_momentum[2], Four_momentum[3] + obj.Four_momentum[3]);
            return temp;
        }

        __device__ FV operator-(FV obj) const {
            FV temp(Four_momentum[0] - obj.Four_momentum[0], Four_momentum[1] - obj.Four_momentum[1], Four_momentum[2] - obj.Four_momentum[2], Four_momentum[3] - obj.Four_momentum[3]);
            return temp;
        }

        __device__ double operator*(FV obj) const {
            double temp = Four_momentum[0]*obj.Four_momentum[0] - Four_momentum[1]*obj.Four_momentum[1] - Four_momentum[2]*obj.Four_momentum[2] - Four_momentum[3]*obj.Four_momentum[3];
            return temp;
        }

        __device__ FV operator*(double scale) const {
            FV temp(scale*Four_momentum[0],scale*Four_momentum[1],scale*Four_momentum[2],scale*Four_momentum[3]);
            return temp;
        }

        __device__ FV operator/(double scale) const {
            FV temp(Four_momentum[0]/scale,Four_momentum[1]/scale,Four_momentum[2]/scale,Four_momentum[3]/scale);
            return temp;
        }

        __device__ FV &operator=(const FV& obj){
            Four_momentum[0] = obj.Four_momentum[0];
            Four_momentum[1] = obj.Four_momentum[1];
            Four_momentum[2] = obj.Four_momentum[2];
            Four_momentum[3] = obj.Four_momentum[3];
            return *this;
        }

};

#endif // CTPWA_FRAMEWORK_MATH_FOUR_VECTOR_CUH
