// Named Lorentz contractions for rank-two tensors. These helpers make every
// lowered index explicit and avoid adding an ambiguous tensor*tensor operator.
#ifndef CTPWA_FRAMEWORK_TENSORS_TENSOR_CONTRACTION_CUH
#define CTPWA_FRAMEWORK_TENSORS_TENSOR_CONTRACTION_CUH

#include "framework/tensors/Tensor.cuh"

namespace ctpwa {

// A^{mu nu} v_nu. New tensor code should prefer this named form to the
// equivalent legacy `tensor * FV` operator when the contracted index matters.
__device__ inline FV contract_second_index(
    const tensor& source,
    const FV& vector)
{
    return source * vector;
}

// Symmetric part A^{(mu nu)}.
__device__ inline tensor symmetrize(const tensor& source)
{
    tensor result;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            result._matrix[mu][nu] = 0.5
                * (source._matrix[mu][nu] + source._matrix[nu][mu]);
        }
    }
    return result;
}

// A^mu_mu = g_{mu nu} A^{mu nu}. The Cartesian metric is diagonal.
__device__ inline double lorentz_trace(const tensor& source)
{
    double result = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        result += metric_sign(mu) * source._matrix[mu][mu];
    }
    return result;
}

// A^{mu nu} B_{mu nu}; both tensor arguments store contravariant components.
__device__ inline double double_contract(
    const tensor& first,
    const tensor& second)
{
    double result = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            result += metric_sign(mu) * metric_sign(nu)
                * first._matrix[mu][nu] * second._matrix[mu][nu];
        }
    }
    return result;
}

// C^{mu nu} = A^{mu alpha} B^nu_alpha
//             = sum_alpha g_alphaalpha A^{mu alpha} B^{nu alpha}.
// The result need not be symmetric even when both inputs are symmetric.
__device__ inline tensor contract_second_indices(
    const tensor& first,
    const tensor& second)
{
    tensor result;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            for (int alpha = 0; alpha < 4; ++alpha) {
                result._matrix[mu][nu] += metric_sign(alpha)
                    * first._matrix[mu][alpha]
                    * second._matrix[nu][alpha];
            }
        }
    }
    return result;
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_TENSORS_TENSOR_CONTRACTION_CUH
