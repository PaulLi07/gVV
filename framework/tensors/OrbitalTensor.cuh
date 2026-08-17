#ifndef CTPWA_FRAMEWORK_TENSORS_ORBITAL_TENSOR_CUH
#define CTPWA_FRAMEWORK_TENSORS_ORBITAL_TENSOR_CUH

#include "framework/tensors/SpinProjector.cuh"

namespace ctpwa {

__device__ inline FV orbital_pwave(
    const FV& parent,
    const FV& relative_momentum)
{
    return transverse_projector(parent) * relative_momentum;
}

__device__ inline tensor orbital_dwave(
    const FV& parent,
    const FV& relative_momentum)
{
    const tensor projector = transverse_projector(parent);
    const FV projected = projector * relative_momentum;
    return tensor(projected, projected)
           - projector * ((projector * relative_momentum) * relative_momentum)
                 / 3.0;
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_TENSORS_ORBITAL_TENSOR_CUH
