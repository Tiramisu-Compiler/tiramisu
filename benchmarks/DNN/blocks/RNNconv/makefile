#===============================================================================
# Copyright 2016-2017 Intel Corporation All Rights Reserved.
#
# The source code,  information  and material  ("Material") contained  herein is
# owned by Intel Corporation or its  suppliers or licensors,  and  title to such
# Material remains with Intel  Corporation or its  suppliers or  licensors.  The
# Material  contains  proprietary  information  of  Intel or  its suppliers  and
# licensors.  The Material is protected by  worldwide copyright  laws and treaty
# provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
# modified, published,  uploaded, posted, transmitted,  distributed or disclosed
# in any way without Intel's prior express written permission.  No license under
# any patent,  copyright or other  intellectual property rights  in the Material
# is granted to  or  conferred  upon  you,  either   expressly,  by implication,
# inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
# property rights must be express and approved by Intel in writing.
#
# Unless otherwise agreed by Intel in writing,  you may not remove or alter this
# notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
# suppliers or licensors in any way.
#===============================================================================

##  Content:
##      Intel(R) Math Kernel Library DNN examples creation and run
##******************************************************************************

DNNC = $(DNNC_S)
#$(DNNC_D)

DNNC_S = s_score_sample
# s_train_sample
#DNNC_D = d_score_sample d_train_sample

help:
	@echo $$'$(WRONG_OPTION)'
	@echo "Intel(R) MKL DNN examples help."
	@echo ""
	@echo "Usage: make Target [Options...] [Example...]"
	@echo ""
	@echo "   Target:"
	@echo "       libia32      - statically linked example for processors that use the IA-32 architecture"
	@echo "       dylibia32    - dynamically linked example for processors that use the IA-32 architecture"
	@echo "       libintel64   - statically linked example for processors that use the Intel(R) 64 architecture"
	@echo "       dylibintel64 - dynamically linked example for processors that use the Intel(R) 64 architecture"
	@echo "       help         - print this help"
	@echo ""
	@echo "   Options:"
	@echo "       interface={lp64|ilp64}"
	@echo "           programming interface for intel64."
	@echo "           Default: lp64."
	@echo "       threading={parallel|sequential}"
	@echo "           Specifies whether to use Intel(R) MKL in the threaded or sequential mode."
	@echo "           Default: parallel."
	@echo "       compiler={intel|gnu|clang}"
	@echo "           Specifies the compiler used to compile the example."
	@echo "           Default: intel."
	@echo "       MKLROOT=<MKL_directory>"
	@echo "           Specifies the location of Intel(R) MKL libraries used to build this example."
	@echo "           Default: the Intel(R) MKL installation directory."
	@echo ""
	@echo "   Example:"
	@echo "       example=example_name1 example_name2 ..."
	@echo "           Specifies the list of required examples."
	@echo "           The supported example list:"
	@echo "               $(DNNC)"
	@echo "           Default: all examples."
	@echo ""
	@echo "Usage examples:"
	@echo ""
	@echo "   make libintel64 example=\"s_score_sample s_train_sample\""
	@echo "       Link 's_score_sample' and ' s_train_sample' examples against parallel static MKL-Intel64."
	@echo ""
	@echo "   make dylibintel64 interface=ilp64 threading=sequential"
	@echo "       Link all the examples against sequential dynamic MKL-Intel64"
	@echo "       using ILP64 interface."
	@echo ""
#------------------------------------------------------------------------------


ifndef example
example = $(DNNC)
endif

TARGET = $(DNNC)

# Defaults
compiler=intel
interface=lp64
threading=parallel

ifneq ($(compiler),intel)
ifneq ($(compiler),gnu)
ifneq ($(compiler),clang)
MSG2+= compiler=$(compiler)
endif
endif
endif

ifneq ($(interface),ilp64)
ifneq ($(interface),lp64)
MSG2+= interface=$(interface)
endif
endif

ifneq ($(threading),parallel)
ifneq ($(threading),sequential)
MSG2+= threading=$(threading)
endif
endif

ifneq ("$(MSG2)","")
WRONG_OPTION=\n\n*** COMMAND LINE ERROR: Wrong value of option(s):  $(MSG2)\n\n
TARGET=help
endif
###

ifdef _IA

ifeq ($(SD),static)
  EXT=a
  RES_EXT=lib
else
  EXT=dylib
  RES_EXT=dylib
endif

ifndef MKLROOT
MKLROOT = ${MKL_PREFIX}
endif
MKL_PATH = $(MKLROOT)/lib
CMPLR_PATH = $(MKLROOT)/../compiler/lib

gnu_or_clang = $(if $(filter gnu,$(compiler)),gcc,$(filter clang,$(compiler)))
ifneq ($(gnu_or_clang),)
  override COMPILER = $(gnu_or_clang)
else
  override COMPILER = icc
endif
IFACE_COMP_PART = intel
IFACE_THREADING_PART = intel
OPTIONS = -w
#-std=c++11

ifeq ($(_IA),ia32)
  ifeq ($(interface),ilp64)
    $(warning  *** ILP64 interface is not available for MKL-IA32)
    $(error Try >make help)
  endif
  ifeq ($(compiler),intel)
      # This option tells the compiler to generate optimized code
      # for Pentium or later processor.
      # If you don't need it, you can remove this option.
    OPTIONS += -mia32
  endif
  IFACE_SUFF=
  M32_64 = -m32 # This option tells compiler to generate code for IA-32 architecture.
else
  IFACE_SUFF=_$(interface)
  M32_64 = -m64 # This option tells compiler to generate code for Intel64 architecture.
endif

IFACE_LIB="$(MKL_PATH)/libmkl_$(IFACE_COMP_PART)$(IFACE_SUFF).$(EXT)"

ifeq ($(interface),ilp64)
  OPTIONS += -DMKL_ILP64
endif

ifeq ($(threading),sequential)
  THREADING_LIB="$(MKL_PATH)/libmkl_sequential.$(EXT)"
  OMP_LIB =
else
  THREADING_LIB="$(MKL_PATH)/libmkl_$(IFACE_THREADING_PART)_thread.$(EXT)"
  OMP_LIB = -L"$(CMPLR_PATH)" -liomp5
endif

RPATH = -Wl,-rpath,$(MKL_PATH) -Wl,-rpath,$(CMPLR_PATH)
CORE_LIB="$(MKL_PATH)/libmkl_core.$(EXT)"

LABEL = $(compiler)$(IFACE_SUFF)_$(threading)_$(_IA)_$(RES_EXT)
RES_DIR = _results/$(LABEL)

ifneq ("$(out_to_screen)","")
OUT_TO_FILE=
else
OUT_TO_FILE=> $(RES_DIR)/$@.res
endif

endif # ifdef _IA

#-------------------------------------------------------------------------------

ia32: libia32 dylibia32
intel64: libintel64 dylibintel64

libia32 lib32:
	@$(MAKE) $(TARGET) --no-print-directory SD=static  _IA=ia32
dylibia32 dylib32:
	@$(MAKE) $(TARGET) --no-print-directory SD=dynamic _IA=ia32
libintel64 libem64t:
	@$(MAKE) $(TARGET) --no-print-directory SD=static  _IA=intel64
dylibintel64 dylibem64t:
	@$(MAKE) $(TARGET) --no-print-directory SD=dynamic _IA=intel64

#-------------------------------------------------------------------------------

$(DNNC):
	@mkdir -p ./$(RES_DIR)
	@echo $$'\n----- Compiling $(LABEL) ----- $@'
	$(COMPILER) $(M32_64) $(OPTIONS) -I"$(MKLROOT)/include" \
	$@.c \
	$(IFACE_LIB) \
	$(THREADING_LIB) \
	$(CORE_LIB) \
	$(RPATH) \
	$(OMP_LIB) -lpthread -lm -o $(RES_DIR)/$@.out
	@echo ----- Execution $(LABEL) ----- $@
	$(RES_DIR)/$@.out

#$(OUT_TO_FILE)

#-------------------------------------------------------------------------------
