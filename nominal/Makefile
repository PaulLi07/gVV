PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR ?= $(PROJECT_ROOT)/build
OBJ_DIR ?= $(BUILD_DIR)/obj
BIN_DIR ?= $(PROJECT_ROOT)/bin
TEST_BIN_DIR ?= $(PROJECT_ROOT)/tests/bin

# These defaults are also exported by config/gvv_env.sh.  Keeping fallbacks
# here allows `make` to work from a clean shell while the project environment
# remains explicit and reproducible when sourced.
GVV_CUDA_ROOT ?= /usr/local/cuda-12
ROOT_PREFIX ?= /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.32.02/x86_64-almalinux9.4-gcc114-opt
NVCC ?= $(GVV_CUDA_ROOT)/bin/nvcc

LIBS=-L$(ROOT_PREFIX)/lib -lCore -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lm -ldl -m64 -lRooFitCore -lRooFit -lrt -lMinuit -lFoam -lMathMore
CFLAGS=-I$(ROOT_PREFIX)/include
INCLUDES=-I$(PROJECT_ROOT)/include
RPATH=-Xlinker -rpath -Xlinker $(ROOT_PREFIX)/lib
CUDA_ARCH=-gencode arch=compute_70,code=sm_70 \
          -gencode arch=compute_80,code=sm_80
COMPILEFLAGS=-O2 -g -std=c++17 $(CUDA_ARCH)
DVAR=-DUSEROOT

COMMON_OBJECTS=$(OBJ_DIR)/kernel.o $(OBJ_DIR)/OmegaPropagator.o $(OBJ_DIR)/GVVSample.o $(OBJ_DIR)/GVVFitParameters.o $(OBJ_DIR)/ModelDefinition.o
FIT_OBJECTS=$(COMMON_OBJECTS) $(OBJ_DIR)/NLL_estimator.o $(OBJ_DIR)/Fit.o
POSTFIT_OBJECTS=$(COMMON_OBJECTS) $(OBJ_DIR)/PostFit.o

.PHONY: all tests clean

all: $(BIN_DIR)/Fit.exe $(BIN_DIR)/PostFit.exe

$(OBJ_DIR) $(BIN_DIR) $(TEST_BIN_DIR):
	mkdir -p $@

$(OBJ_DIR)/kernel.o: src/kernel.cu include/kernel.h include/GVVAmplitude.h include/GVVModel.h include/Dynamics.h | $(OBJ_DIR)
	$(NVCC) $(LIBS) $(CFLAGS) -c -w $< $(INCLUDES) $(DVAR) $(COMPILEFLAGS) -o $@

$(OBJ_DIR)/OmegaPropagator.o: src/OmegaPropagator.cu include/OmegaPropagator.h include/Dynamics.h | $(OBJ_DIR)
	$(NVCC) $(LIBS) $(CFLAGS) -c -w $< $(INCLUDES) $(DVAR) $(COMPILEFLAGS) -o $@

$(OBJ_DIR)/GVVSample.o: src/GVVSample.cu include/GVVSample.h include/kernel.h | $(OBJ_DIR)
	$(NVCC) $(LIBS) $(CFLAGS) -c -w $< $(INCLUDES) $(DVAR) $(COMPILEFLAGS) -o $@

$(OBJ_DIR)/GVVFitParameters.o: src/GVVFitParameters.cu include/GVVFitParameters.h include/GVVModel.h | $(OBJ_DIR)
	$(NVCC) $(LIBS) $(CFLAGS) -c -w $< $(INCLUDES) $(DVAR) $(COMPILEFLAGS) -o $@

$(OBJ_DIR)/ModelDefinition.o: src/ModelDefinition.cpp include/framework/ModelDefinition.h | $(OBJ_DIR)
	$(NVCC) $(LIBS) $(CFLAGS) -c -w $< $(INCLUDES) $(DVAR) $(COMPILEFLAGS) -o $@

$(OBJ_DIR)/NLL_estimator.o: src/NLL_estimator.cu include/NLL_estimator.h include/GVVSample.h include/GVVModel.h include/Dynamics.h | $(OBJ_DIR)
	$(NVCC) $(LIBS) $(CFLAGS) -c -w $< $(INCLUDES) $(DVAR) $(COMPILEFLAGS) -o $@

$(OBJ_DIR)/Fit.o: src/Fit.cu include/Minuit.h include/NLL_estimator.h include/GVVModel.h include/GVVFitParameters.h | $(OBJ_DIR)
	$(NVCC) $(LIBS) $(CFLAGS) -c -w $< $(INCLUDES) $(DVAR) $(COMPILEFLAGS) -o $@

$(OBJ_DIR)/PostFit.o: src/PostFit.cu include/GVVFitParameters.h include/GVVSample.h include/kernel.h | $(OBJ_DIR)
	$(NVCC) $(LIBS) $(CFLAGS) -c -w $< $(INCLUDES) $(DVAR) $(COMPILEFLAGS) -o $@

$(BIN_DIR)/Fit.exe: $(FIT_OBJECTS) | $(BIN_DIR)
	$(NVCC) -w $(FIT_OBJECTS) $(LIBS) $(RPATH) $(CFLAGS) $(DVAR) $(COMPILEFLAGS) -o $@

$(BIN_DIR)/PostFit.exe: $(POSTFIT_OBJECTS) | $(BIN_DIR)
	$(NVCC) -w $(POSTFIT_OBJECTS) $(LIBS) $(RPATH) $(CFLAGS) $(DVAR) $(COMPILEFLAGS) -o $@

tests: $(TEST_BIN_DIR)/test_dynamics.exe $(TEST_BIN_DIR)/test_gvv_amplitude.exe $(TEST_BIN_DIR)/test_gvv_model.exe $(TEST_BIN_DIR)/test_gvv_fit_parameters.exe $(TEST_BIN_DIR)/test_model_definition.exe

$(TEST_BIN_DIR)/test_dynamics.exe: tests/test_dynamics.cu include/Dynamics.h | $(TEST_BIN_DIR)
	$(NVCC) -w $< $(INCLUDES) $(COMPILEFLAGS) -o $@

$(TEST_BIN_DIR)/test_gvv_amplitude.exe: tests/test_gvv_amplitude.cu | $(TEST_BIN_DIR)
	$(NVCC) -w $< $(INCLUDES) $(COMPILEFLAGS) -o $@

$(TEST_BIN_DIR)/test_gvv_model.exe: tests/test_gvv_model.cu include/GVVModel.h include/Dynamics.h $(OBJ_DIR)/OmegaPropagator.o | $(TEST_BIN_DIR)
	$(NVCC) -w $< $(OBJ_DIR)/OmegaPropagator.o $(INCLUDES) $(COMPILEFLAGS) -o $@

$(TEST_BIN_DIR)/test_gvv_fit_parameters.exe: tests/test_gvv_fit_parameters.cu tests/fit_result_legacy_fixture.txt tests/fit_result_machine_fixture.txt $(OBJ_DIR)/GVVFitParameters.o | $(TEST_BIN_DIR)
	$(NVCC) -w $< $(OBJ_DIR)/GVVFitParameters.o $(INCLUDES) $(COMPILEFLAGS) -o $@

$(TEST_BIN_DIR)/test_model_definition.exe: tests/test_model_definition.cpp $(OBJ_DIR)/ModelDefinition.o | $(TEST_BIN_DIR)
	$(NVCC) -w $< $(OBJ_DIR)/ModelDefinition.o $(INCLUDES) $(COMPILEFLAGS) -o $@

clean:
	rm -f $(FIT_OBJECTS) $(POSTFIT_OBJECTS) $(BIN_DIR)/Fit.exe $(BIN_DIR)/PostFit.exe $(TEST_BIN_DIR)/*.exe
