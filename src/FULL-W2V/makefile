include ../../common.inc

# FULL-W2V Compiling Routines
FULL-W2V: FULL-W2V-driver.cu \
			FULL-W2V-kernels.cu \
			improved_driver_functions.cpp \
			legacy_driver_functions.cpp \
			expTable.h \
			safecuda.h
	$(GPUCC) FULL-W2V-driver.cu -o FULL-W2V $(NVIDIA_FLAGS)
gdb_FULL-W2V: FULL-W2V-driver.cu \
				FULL-W2V-kernels.cu \
				improved_driver_functions.cpp \
				legacy_driver_functions.cpp \
				expTable.h \
				safecuda.h
	$(GPUCC) FULL-W2V-driver.cu -o gdb_FULL-W2V $(NVIDIA_DEBUG_FLAGS)
REGISTER: REGISTER-driver.cu \
			REGISTER-kernels.cu \
			improved_driver_functions.cpp \
			legacy_driver_functions.cpp \
			expTable.h \
			safecuda.h
	$(GPUCC) REGISTER-driver.cu -o REGISTER $(NVIDIA_FLAGS)
gdb_REGISTER: REGISTER-driver.cu \
			REGISTER-kernels.cu \
			improved_driver_functions.cpp \
			legacy_driver_functions.cpp \
			expTable.h \
			safecuda.h
	$(GPUCC) REGISTER-driver.cu -o REGISTER $(NVIDIA_DEBUG_FLAGS)

# FULL-W2V SUBROUTINES
expTable: generate_expTable_h.cpp
	$(CPU_GCC) generate_expTable_h.cpp -o geT_h -lm $(C_FLAGS)
	./geT_h 1000 > expTable.h
	rm geT_h
expTable-no-cuda: generate_expTable_h.cpp
	$(CPU_GCC) generate_expTable_h.cpp -o geT_h -lm $(C_FLAGS)
	./geT_h 1000 -no-cuda > expTable.h
	rm geT_h

