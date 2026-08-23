// Host catalogue for complete GVV Waves. Model and propagator compilation are
// deliberately kept out of this registration unit.
#include "process/WaveRegistry.cuh"

#include <stdexcept>

const std::vector<GVVWaveMetadata>& gvv_wave_registry()
{
    // This is the only host registration point for complete GVV waves. The
    // matching device dispatch is the single gvv_wave_tensor function in
    // WaveRegistry.cuh; individual formulae stay in process/waves/.
    static const std::vector<GVVWaveMetadata> registry = {
        {"gvv.scalar_00", "0++", "0^{++}(00)", "scalar", GVV_SCALAR_00},
        {"gvv.scalar_22", "0++", "0^{++}(22)", "scalar", GVV_SCALAR_22},

        {"gvv.pseudoscalar_11", "0-+", "0^{-+}(11)",
         "pseudoscalar", GVV_PSEUDOSCALAR_11},

        {"gvv.tensor_02_u1", "2++", "2^{++}:U^{(1)}_{02}",
         "scalar", GVV_TENSOR_02_U1},
        {"gvv.tensor_02_u2", "2++", "2^{++}:U^{(2)}_{02}",
         "scalar", GVV_TENSOR_02_U2},
        {"gvv.tensor_02_u3", "2++", "2^{++}:U^{(3)}_{02}",
         "scalar", GVV_TENSOR_02_U3},

        {"gvv.tensor_20_u1", "2++", "2^{++}:U^{(1)}_{20}",
         "scalar", GVV_TENSOR_20_U1},
        {"gvv.tensor_20_u2", "2++", "2^{++}:U^{(2)}_{20}",
         "scalar", GVV_TENSOR_20_U2},
        {"gvv.tensor_20_u3", "2++", "2^{++}:U^{(3)}_{20}",
         "scalar", GVV_TENSOR_20_U3},

        {"gvv.tensor_22_u1", "2++", "2^{++}:U^{(1)}_{22}",
         "scalar", GVV_TENSOR_22_U1},
        {"gvv.tensor_22_u2", "2++", "2^{++}:U^{(2)}_{22}",
         "scalar", GVV_TENSOR_22_U2},
        {"gvv.tensor_22_u3", "2++", "2^{++}:U^{(3)}_{22}",
         "scalar", GVV_TENSOR_22_U3},

        {"gvv.tensor_42_u1", "2++", "2^{++}:U^{(1)}_{42}",
         "scalar", GVV_TENSOR_42_U1},
        {"gvv.tensor_42_u2", "2++", "2^{++}:U^{(2)}_{42}",
         "scalar", GVV_TENSOR_42_U2},
        {"gvv.tensor_42_u3", "2++", "2^{++}:U^{(3)}_{42}",
         "scalar", GVV_TENSOR_42_U3}
    };
    return registry;
}

const GVVWaveMetadata& gvv_registered_wave(const std::string& id)
{
    for (const GVVWaveMetadata& wave : gvv_wave_registry()) {
        if (wave.id == id) {
            return wave;
        }
    }
    throw std::runtime_error(
        "model references unregistered GVV wave '" + id + "'");
}
