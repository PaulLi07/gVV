// Complete GVV process compiler. It resolves active Term dependencies, calls
// the independent Wave and propagator registries, and builds dense GPU slots.
#ifndef CTPWA_PROCESS_MODEL_COMPILER_H
#define CTPWA_PROCESS_MODEL_COMPILER_H

#include "process/ProcessModel.h"

GVVCompiledModel gvv_compile_model(
    const ctpwa::ModelDefinition& definition);

GVVCompiledModel gvv_load_compiled_model(const std::string& file_name);

// Bump this contract identifier whenever a change alters the numerical meaning
// of a registered Wave, Resonance propagator, Term assembly, or fit binding.
const char* gvv_amplitude_implementation_signature();

// Preserve the legacy key for models without an explicit resolution setting.
std::string gvv_model_implementation_signature(
    const ctpwa::ModelDefinition& definition);

// Combined compatibility key for one declarative model and this GVV amplitude
// implementation. Fit and Post Calculation must agree on this value.
std::string gvv_model_signature(
    const ctpwa::ModelDefinition& definition);

#endif // CTPWA_PROCESS_MODEL_COMPILER_H
