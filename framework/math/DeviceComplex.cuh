#ifndef DEVICE_COMPLEX_H
#define DEVICE_COMPLEX_H

//======================================================================
//The complex number class used in GPU
//======================================================================
#include <cmath>
#include <stdio.h>
#include <iostream>


class DeviceComplex {
public:
    double real;
    double imag;
     __host__ __device__ DeviceComplex(double r = 0, double i = 0) : real(r), imag(i) {}
    // Simple operators complex - complex
     __host__ __device__ DeviceComplex operator+(const DeviceComplex& other) const {
        return DeviceComplex(real + other.real, imag + other.imag);
    }
     __host__ __device__ DeviceComplex operator-(const DeviceComplex& other) const {
        return DeviceComplex(real - other.real, imag - other.imag);
    }
     __host__ __device__ DeviceComplex operator*(const DeviceComplex& other) const {
        return DeviceComplex(real * other.real - imag * other.imag, real * other.imag + imag * other.real);
    }
     __host__ __device__ DeviceComplex operator/(const DeviceComplex& other) const {
        double denom = other.real * other.real + other.imag * other.imag;
        return DeviceComplex((real * other.real + imag * other.imag) / denom, (imag * other.real - real * other.imag) / denom);
    }

    // Simple operators complex - double
     __host__ __device__ DeviceComplex operator *(double c) const {
        return DeviceComplex(real * c, imag * c);
    }
     __host__ __device__ DeviceComplex operator +(double c) const {
        return DeviceComplex(real + c, imag);
    }
     __host__ __device__ DeviceComplex operator /(double c) const {
        return DeviceComplex(real / c, imag / c);
    }
     __host__ __device__ DeviceComplex operator -(double c) const {
        return DeviceComplex(real - c, imag);
    }

    // Simple operators double - complex
     __host__ __device__ friend DeviceComplex operator *(double c, const DeviceComplex & other) {
        return DeviceComplex(c * other.real, c * other.imag);
    }
     __host__ __device__ friend DeviceComplex operator +(double c, const DeviceComplex & other) {
        return DeviceComplex(c + other.real, other.imag);
    }
     __host__ __device__ friend DeviceComplex operator /(double c, const DeviceComplex & other) {
        double denom = other.real * other.real + other.imag * other.imag;
        return DeviceComplex(c * other.real, -c * other.imag) / denom;
    }
     __host__ __device__ friend DeviceComplex operator -(double c, const DeviceComplex & other) {
        return DeviceComplex(c - other.real, -other.imag);
    }

     __host__ __device__ double rho() const {
        return sqrt(real * real + imag * imag);
    }
     __host__ __device__ double rho2() const {
        return real * real + imag * imag;
    }
     __host__ __device__ double phi() const {
        double theta = acos(real/sqrt(real * real + imag * imag));
        if(imag>0){return theta;}
        else{return theta+3.1415927;}
     }

     __host__ __device__ DeviceComplex reciprocal() const {
        double denom = real * real + imag * imag;
        return DeviceComplex(real / denom, -imag / denom);
    }
     __host__ __device__ DeviceComplex conjugate() const {
        return DeviceComplex(real, -imag);
    }
     __host__ __device__ void print() const {
        printf("%f + %fi\n", real, imag);
    }
     __host__ __device__ DeviceComplex& operator=(const DeviceComplex& other) {
        if (this != &other) {
            real = other.real;
            imag = other.imag;
        }
        return *this;
    }
    
};

#endif // DEVICE_COMPLEX_H
