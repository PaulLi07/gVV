// Independent CPU quadrature checks for the complex mass convolution.
#include "core/physics/OmegaResolution.cuh"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

using Complex = std::complex<double>;
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
Complex as_complex(DeviceComplex value) { return {value.real, value.imag}; }

// Composite Simpson in Gaussian pull, with independent std::complex BW
// arithmetic. Dense sampling does not share GL nodes or knot splitting with
// the production algorithm. Doubling N verifies reference convergence.
Complex reference(double mass, double sigma,
                  ctpwa::TabulatedFunctionView table, int n, bool intensity = false)
{
    const double a = std::max(-9.0, -mass / sigma), b = 9.0;
    const double h = (b - a) / n;
    Complex sum(0.0, 0.0);
    for (int i = 0; i <= n; ++i) {
        const double z = a + i * h;
        const double m = mass + sigma * z;
        const Complex bw = 1.0 / Complex(GVV_OMEGA_MASS * GVV_OMEGA_MASS - m * m,
            -GVV_OMEGA_MASS * table.interpolate_clamped(m * m));
        const double weight = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        sum += weight * std::exp(-0.5 * z * z)
            * (intensity ? Complex(std::norm(bw), 0.0) : bw);
    }
    return sum * h / (3.0 * std::sqrt(2.0 * std::acos(-1.0)));
}
int main()
{
    try {
        OmegaWidthTable owner;
        owner.Build(); // Host only: no CUDA device or upload required.
        const auto table = owner.HostView();
        double maximum_error = 0.0, maximum_reference_error = 0.0;
        for (double mass : {0.40, 0.415, 0.70, 0.765, GVV_OMEGA_MASS,
                            0.79, 0.805, 0.88, 1.20}) {
            const auto original = gvv_omega_propagator(mass * mass, table);
            const auto zero = gvv_smeared_omega_propagator(mass * mass, table, 0.0);
            require(original.real == zero.real && original.imag == zero.imag,
                    "sigma=0 did not preserve the exact legacy propagator");
            const auto sub_ulp = gvv_smeared_omega_propagator(mass * mass, table, 1.e-30);
            require(sub_ulp.real == original.real && sub_ulp.imag == original.imag,
                    "sub-ulp sigma did not recover the legacy limit");
            const auto tiny = as_complex(gvv_smeared_omega_propagator(mass * mass, table, 1.e-7));
            require(std::abs(tiny - as_complex(original))
                    < 2.e-7 * std::max(1.0, std::abs(as_complex(original))),
                    "small-sigma limit failed");
            for (double sigma : {0.0001, 0.001, 0.005, 0.015, 0.03, 0.05}) {
                const auto coarse = reference(mass, sigma, table, 32768);
                const auto fine = reference(mass, sigma, table, 65536);
                const auto actual = as_complex(gvv_smeared_omega_propagator(mass * mass, table, sigma));
                const double scale = std::max(1.0, std::abs(fine));
                const double ref_error = std::abs(coarse - fine) / scale;
                const double error = std::abs(actual - fine) / scale;
                maximum_reference_error = std::max(maximum_reference_error, ref_error);
                maximum_error = std::max(maximum_error, error);
                require(ref_error < 2.e-8, "independent Simpson reference did not converge");
                require(error < 2.e-8, "complex convolution disagrees with independent integral");
            }
        }
        const double m = GVV_OMEGA_MASS, sigma = 0.005;
        const auto complex_convolution = as_complex(gvv_smeared_omega_propagator(m * m, table, sigma));
        const double intensity_convolution = reference(m, sigma, table, 65536, true).real();
        require(std::fabs(std::norm(complex_convolution) - intensity_convolution)
                    > 0.1 * intensity_convolution,
                "test must distinguish amplitude convolution from intensity convolution");
        // Check derivatives on both sides of tiny sigma displacements. The
        // independent reference bounds any numerical panel-transition noise.
        for (double center : {0.001, 0.00434, 0.005, 0.03}) {
            const double h = center * 1.e-4;
            auto production = [&](double s) {
                return as_complex(gvv_smeared_omega_propagator(0.781 * 0.781, table, s));
            };
            const auto d1 = (production(center+h)-production(center-h))/(2*h);
            const auto d2 = (production(center+h/2)-production(center-h/2))/h;
            require(std::abs(d1-d2) < 1.e-6 * std::max(1.0, std::abs(d2)),
                    "sigma derivative is numerically unstable");
        }
        std::cout << "Omega complex convolution CPU tests passed; max relative error="
                  << maximum_error << ", reference convergence=" << maximum_reference_error << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
