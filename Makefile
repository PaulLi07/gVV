# gVV build entry point. framework objects are process-neutral; process
# objects implement psi(2S)->gamma omega omega; app/Fit.cu is the glue layer.
# CUDA/ROOT locations and build directories may be overridden by the caller.
PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR ?= $(PROJECT_ROOT)/build
OBJ_DIR ?= $(BUILD_DIR)/obj
BIN_DIR ?= $(PROJECT_ROOT)/bin
TEST_BIN_DIR ?= $(BUILD_DIR)/tests

# Reproducible default toolchain used on the IHEP AlmaLinux GPU nodes.
GVV_CUDA_ROOT ?= /usr/local/cuda-12
ROOT_PREFIX ?= /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.32.02/x86_64-almalinux9.4-gcc114-opt
NVCC ?= $(GVV_CUDA_ROOT)/bin/nvcc

ROOT_LIBS = -L$(ROOT_PREFIX)/lib -lCore -lRIO -lNet -lHist -lGraf \
	-lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics \
	-lMathCore -lThread -lm -ldl -m64 -lRooFitCore -lRooFit -lrt \
	-lMinuit -lFoam -lMathMore
ROOT_INCLUDES = -I$(ROOT_PREFIX)/include
PROJECT_INCLUDES = -I$(PROJECT_ROOT)
RPATH = -Xlinker -rpath -Xlinker $(ROOT_PREFIX)/lib
CUDA_ARCH = -gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_80,code=sm_80
COMPILE_FLAGS = -O2 -g -std=c++17 $(CUDA_ARCH)
DEPENDENCY_FLAGS = -MMD -MP

# Keep the generic/process split visible at link time. app/Fit.o is the only
# executable-specific object.
FRAMEWORK_OBJECTS = \
	$(OBJ_DIR)/Model.o \
	$(OBJ_DIR)/FitConfig.o \
	$(OBJ_DIR)/FitEngine.o \
	$(OBJ_DIR)/FitOutput.o \
	$(OBJ_DIR)/FitState.o
PROCESS_OBJECTS = \
	$(OBJ_DIR)/WaveRegistry.o \
	$(OBJ_DIR)/TermEvaluator.o \
	$(OBJ_DIR)/OmegaWidthTable.o \
	$(OBJ_DIR)/SampleLoader.o \
	$(OBJ_DIR)/ParameterMapping.o \
	$(OBJ_DIR)/FitLikelihood.o \
	$(OBJ_DIR)/ProjectionWriter.o
FIT_OBJECTS = $(FRAMEWORK_OBJECTS) $(PROCESS_OBJECTS) $(OBJ_DIR)/Fit.o
POST_OBJECTS = \
	$(OBJ_DIR)/Model.o \
	$(OBJ_DIR)/FitState.o \
	$(OBJ_DIR)/WaveRegistry.o \
	$(OBJ_DIR)/TermEvaluator.o \
	$(OBJ_DIR)/OmegaWidthTable.o \
	$(OBJ_DIR)/SampleLoader.o \
	$(OBJ_DIR)/ParameterMapping.o \
	$(OBJ_DIR)/ComponentEvaluator.o \
	$(OBJ_DIR)/PostCalculation.o
ALL_OBJECTS = $(sort $(FIT_OBJECTS) $(POST_OBJECTS))

TESTS = \
	$(TEST_BIN_DIR)/test_dynamics.exe \
	$(TEST_BIN_DIR)/test_gvv_amplitude.exe \
	$(TEST_BIN_DIR)/test_propagator_registry.exe \
	$(TEST_BIN_DIR)/test_fit_parameters.exe \
	$(TEST_BIN_DIR)/test_fit_config.exe \
	$(TEST_BIN_DIR)/test_fit_output.exe \
	$(TEST_BIN_DIR)/test_fit_state.exe \
	$(TEST_BIN_DIR)/test_fit_engine.exe \
	$(TEST_BIN_DIR)/test_model.exe \
	$(TEST_BIN_DIR)/test_wave_registry.exe \
	$(TEST_BIN_DIR)/test_likelihood.exe
GPU_TESTS = \
	$(TEST_BIN_DIR)/test_gvv_wave_numerics.exe \
	$(TEST_BIN_DIR)/test_intensity_equivalence.exe
DEPENDENCY_FILES = \
	$(ALL_OBJECTS:.o=.d) \
	$(TESTS:.exe=.d) \
	$(GPU_TESTS:.exe=.d)

# Dependency generation belongs only to compilation targets, not the final
# executable link step.
$(ALL_OBJECTS) $(TESTS) $(GPU_TESTS): private COMPILE_FLAGS += $(DEPENDENCY_FLAGS)

.PHONY: all fit post tests check gpu-tests check-gpu clean

all: fit

fit: $(BIN_DIR)/Fit.exe

post: $(BIN_DIR)/Post.exe

$(OBJ_DIR) $(BIN_DIR) $(TEST_BIN_DIR):
	mkdir -p $@

$(OBJ_DIR)/Model.o: framework/model/Model.cpp framework/model/Model.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/FitConfig.o: framework/fit/FitConfig.cpp framework/fit/FitConfig.h framework/fit/FitEngine.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/FitEngine.o: framework/fit/FitEngine.cpp framework/fit/FitEngine.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/FitOutput.o: framework/fit/FitOutput.cpp framework/fit/FitOutput.h framework/fit/FitEngine.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/FitState.o: framework/fit/FitState.cpp framework/fit/FitState.h framework/fit/FitEngine.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/WaveRegistry.o: process/WaveRegistry.cu process/WaveRegistry.cuh | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/TermEvaluator.o: process/TermEvaluator.cu process/TermEvaluator.cuh process/ProcessAmplitude.cuh | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/OmegaWidthTable.o: process/OmegaWidthTable.cu process/OmegaWidthTable.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/SampleLoader.o: process/SampleLoader.cu process/SampleLoader.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/ParameterMapping.o: process/ParameterMapping.cu process/ParameterMapping.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/FitLikelihood.o: process/FitLikelihood.cu process/FitLikelihood.h framework/likelihood/Likelihood.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/ProjectionWriter.o: process/ProjectionWriter.cu process/ProjectionWriter.h process/FitLikelihood.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/Fit.o: app/Fit.cu framework/fit/FitConfig.h framework/fit/FitEngine.h framework/fit/FitOutput.h framework/fit/FitState.h process/FitLikelihood.h process/ParameterMapping.h process/ProjectionWriter.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(BIN_DIR)/Fit.exe: $(FIT_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(FIT_OBJECTS) $(ROOT_LIBS) $(RPATH) $(ROOT_INCLUDES) \
		$(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/ComponentEvaluator.o: post/calculation/ComponentEvaluator.cu post/calculation/ComponentEvaluator.h process/TermEvaluator.cuh process/ParameterMapping.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(OBJ_DIR)/PostCalculation.o: post/calculation/PostCalculation.cu post/calculation/ComponentEvaluator.h framework/fit/FitState.h | $(OBJ_DIR)
	$(NVCC) $(ROOT_INCLUDES) -c $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(BIN_DIR)/Post.exe: $(POST_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(POST_OBJECTS) $(ROOT_LIBS) $(RPATH) $(ROOT_INCLUDES) \
		$(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_dynamics.exe: tests/test_dynamics.cu | $(TEST_BIN_DIR)
	$(NVCC) $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_gvv_amplitude.exe: tests/test_gvv_amplitude.cu process/ProcessAmplitude.cuh | $(TEST_BIN_DIR)
	$(NVCC) $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_propagator_registry.exe: tests/test_propagator_registry.cu $(OBJ_DIR)/OmegaWidthTable.o | $(TEST_BIN_DIR)
	$(NVCC) $< $(OBJ_DIR)/OmegaWidthTable.o $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_fit_parameters.exe: tests/test_fit_parameters.cu $(OBJ_DIR)/ParameterMapping.o $(OBJ_DIR)/WaveRegistry.o $(OBJ_DIR)/Model.o | $(TEST_BIN_DIR)
	$(NVCC) $< $(OBJ_DIR)/ParameterMapping.o $(OBJ_DIR)/WaveRegistry.o \
		$(OBJ_DIR)/Model.o $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_fit_config.exe: tests/test_fit_config.cpp $(OBJ_DIR)/FitConfig.o | $(TEST_BIN_DIR)
	$(NVCC) $< $(OBJ_DIR)/FitConfig.o $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_fit_output.exe: tests/test_fit_output.cpp $(OBJ_DIR)/FitOutput.o | $(TEST_BIN_DIR)
	$(NVCC) $< $(OBJ_DIR)/FitOutput.o $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_fit_state.exe: tests/test_fit_state.cpp $(OBJ_DIR)/FitState.o | $(TEST_BIN_DIR)
	$(NVCC) $< $(OBJ_DIR)/FitState.o $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_fit_engine.exe: tests/test_fit_engine.cpp $(OBJ_DIR)/FitEngine.o | $(TEST_BIN_DIR)
	$(NVCC) $< $(OBJ_DIR)/FitEngine.o $(ROOT_LIBS) $(RPATH) \
		$(ROOT_INCLUDES) $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_model.exe: tests/test_model.cpp $(OBJ_DIR)/Model.o | $(TEST_BIN_DIR)
	$(NVCC) $< $(OBJ_DIR)/Model.o $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_wave_registry.exe: tests/test_wave_registry.cu $(OBJ_DIR)/WaveRegistry.o $(OBJ_DIR)/Model.o | $(TEST_BIN_DIR)
	$(NVCC) $< $(OBJ_DIR)/WaveRegistry.o $(OBJ_DIR)/Model.o \
		$(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_likelihood.exe: tests/test_likelihood.cpp | $(TEST_BIN_DIR)
	$(NVCC) $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_gvv_wave_numerics.exe: tests/test_gvv_wave_numerics.cu | $(TEST_BIN_DIR)
	$(NVCC) $< $(PROJECT_INCLUDES) $(COMPILE_FLAGS) -o $@

$(TEST_BIN_DIR)/test_intensity_equivalence.exe: tests/test_intensity_equivalence.cu $(OBJ_DIR)/TermEvaluator.o | $(TEST_BIN_DIR)
	$(NVCC) $< $(OBJ_DIR)/TermEvaluator.o $(PROJECT_INCLUDES) \
		$(COMPILE_FLAGS) -o $@

tests: $(TESTS)

# Tests are deliberately ordinary executables so each returns a clear shell
# status and can also be run individually while developing one module.
check: tests
	@for test in $(TESTS); do $$test || exit $$?; done

# Runtime CUDA checks are explicit because login nodes may provide nvcc but no
# device. They are never part of the ordinary host/unit-test target.
gpu-tests: $(GPU_TESTS)

check-gpu: gpu-tests
	@for test in $(GPU_TESTS); do $$test || exit $$?; done

clean:
	rm -f $(ALL_OBJECTS) $(BIN_DIR)/Fit.exe \
		$(BIN_DIR)/Post.exe $(TESTS) $(GPU_TESTS) $(DEPENDENCY_FILES)

# Include auto-generated header dependencies when they exist. This keeps an
# incremental build correct after editing a nested Wave/tensor header.
-include $(DEPENDENCY_FILES)
