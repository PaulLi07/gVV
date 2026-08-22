// Complete GVV process compiler. It resolves active Term dependencies, calls
// the independent Wave and propagator registries, and builds dense GPU slots.
#ifndef CTPWA_PROCESS_MODEL_COMPILER_H
#define CTPWA_PROCESS_MODEL_COMPILER_H

#include "process/ProcessModel.h"

GVVCompiledModel gvv_compile_model(
    const ctpwa::ModelDefinition& definition);

GVVCompiledModel gvv_load_compiled_model(const std::string& file_name);

#endif // CTPWA_PROCESS_MODEL_COMPILER_H
