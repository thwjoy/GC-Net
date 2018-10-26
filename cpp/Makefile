# Define the compiler
CC := g++

#Use GPU Implementation?
USE_GPU := 1

# Read Tensorflow paths
TF_INC := $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_CFLAGS=$(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') 

# Is the Tensorflow version >= 1.4?
TF_VERSION_GTE_1_4 := $(shell expr `python3 -c 'import tensorflow as tf; print(tf.__version__)' | cut -f1,2 -d.` \>= 1.4)

# Flags required for all cases
CFLAGS := -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared -fPIC -I$(TF_INC) -O2 

# Set a special flag if we are on macOS
ifeq ($(shell uname -s), Darwin)
	CFLAGS += -undefined dynamic_lookup
endif

# Set some more flags if the Tensorflow version is >= 1.4
ifeq ($(TF_VERSION_GTE_1_4), 1)
    CFLAGS += -I$(TF_INC)/external/nsync/public
	LDFLAGS := -L$(TF_LIB) -ltensorflow_framework -L/usr/local/cuda/lib64 -lcuda -lcudart
else
	LDFLAGS :=
endif

# Define build targets
.PHONY: all clean


all: four_d_concat.so

four_d_concat.so: four_d_concat_gpu.o four_d_concat.cc 
	g++ $(CFLAGS) -o four_d_concat.so four_d_concat.cc four_d_concat_gpu.o $(LDFLAGS) -D RUN_GPU=$(USE_GPU)

four_d_concat_gpu.o: four_d_concat.cu 
	nvcc -std=c++11 -c -o four_d_concat_gpu.o four_d_concat.cu $(TF_CFLAGS) -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D RUN_GPU=$(USE_GPU)

clean:
	$(RM) four_d_concat.so four_d_concat_gpu.o
