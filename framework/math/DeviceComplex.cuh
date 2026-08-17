// Minimal double-precision complex type shared by host and CUDA device code.
// It exists because std::complex is not a portable device-side interface in
// the CUDA toolchain used by this project.
#ifndef CTPWA_FRAMEWORK_MATH_DEVICE_COMPLEX_CUH
#define CTPWA_FRAMEWORK_MATH_DEVICE_COMPLEX_CUH

//======================================================================
//The complex number class used in GPU
//======================================================================
#include <cmath>
#include <cstdio>


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
     // Principal direction represented on [0, 2*pi). atan2 also gives a
     // defined zero for 0+0i on the supported host/device math libraries.
     __host__ __device__ double phi() const {
        constexpr double two_pi = 6.28318530717958647692;
        const double angle = atan2(imag, real);
        return angle < 0.0 ? angle + two_pi : angle;
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

#endif // CTPWA_FRAMEWORK_MATH_DEVICE_COMPLEX_CUH
