// Transverse spin projectors for the (+---) metric used by FV and tensor.
#ifndef CTPWA_FRAMEWORK_TENSORS_SPIN_PROJECTOR_CUH
#define CTPWA_FRAMEWORK_TENSORS_SPIN_PROJECTOR_CUH

#include "framework/tensors/TensorContraction.cuh"

namespace ctpwa {

__device__ inline tensor transverse_projector(const FV& momentum)
{
    return tensor::Gnormal()
           - tensor(momentum, momentum) / (momentum * momentum);
}

// Apply the spin-two polarization projector to an arbitrary rank-two source:
//
//   result^{mu nu} = P^(2)^{mu nu}_{rho sigma} source^{rho sigma}.
//
// The rank-four projector is deliberately not materialized. The source is
// symmetrized, projected transverse to momentum in both indices, and made
// traceless in the three-dimensional transverse subspace.
__device__ inline tensor spin2_project(
    const FV& momentum,
    const tensor& source)
{
    const double momentum2 = momentum * momentum;
    const tensor symmetric_source = symmetrize(source);
    const FV longitudinal = contract_second_index(
        symmetric_source, momentum);
    const double double_longitudinal = momentum * longitudinal;

    const tensor transverse_source = symmetric_source
        - (tensor(momentum, longitudinal)
           + tensor(longitudinal, momentum)) / momentum2
        + tensor(momentum, momentum)
              * (double_longitudinal / (momentum2 * momentum2));

    const tensor transverse_metric = transverse_projector(momentum);
    return transverse_source
           - transverse_metric * (lorentz_trace(transverse_source) / 3.0);
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_TENSORS_SPIN_PROJECTOR_CUH
