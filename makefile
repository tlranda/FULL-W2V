.PHONY: clean cleaner \
        FULL-W2V gdb_FULL-W2V REGISTER gdb_REGISTER expTable expTable-no-cuda \
        pWord2Vec gdb_pWord2Vec pWord2Vec_mpi \
        pSGNScc gdb_pSGNScc \
        accSGNS gdb_accSGNS \
        wombat wombat_cpu \

# Consistency
include common.inc

# MASTER ROUTINES
all: expTable FULL-W2V REGISTER pWord2Vec pSGNScc accSGNS wombat
clean:
	rm -f FULL-W2V_$(HOSTNAME) gdb_FULL-W2V_$(HOSTNAME) REGISTER_$(HOSTNAME) gdb_REGISTER_$(HOSTNAME);
	rm -f pWord2Vec_$(HOSTNAME) gdb_pWord2Vec_$(HOSTNAME) pWord2Vec_mpi_$(HOSTNAME);
	rm -f pSGNScc_$(HOSTNAME) gdb_pSGNScc_$(HOSTNAME) pSGNScc_cpu_time;
	rm -f accSGNS_$(HOSTNAME) gdb_accSGNS_$(HOSTNAME) accSGNS_gpu_time;
	rm -f wombat_$(HOSTNAME) wombat_cpu_$(HOSTNAME) code/wombat/kernels.o;
cleaner:
	rm -f FULL-W2V_* gdb_FULL-W2V_* REGISTER_* gdb_REGISTER_*;
	rm -f pWord2Vec_* gdb_pWord2Vec_*;
	rm -f pSGNScc_* gdb_PSGNScc_*;
	rm -f accSGNS_* gdb_accSGNS_*;
	rm -f wombat_* code/wombat/kernels.o;

# All other routines opt to makefiles in code/$(IMPLEMENTATION), where $(IMPLEMENTATION) is given in the rule name
# FULL-W2V GROUP
FULL-W2V:
	cd src/FULL-W2V && make FULL-W2V;
	mv src/FULL-W2V/FULL-W2V ./FULL-W2V_$(HOSTNAME);
gdb_FULL-W2V:
	cd src/FULL-W2V && make gdb_FULL-W2V;
	mv src/FULL-W2V/gdb_FULL-W2V ./gdb_FULL-W2V_$(HOSTNAME);
REGISTER:
	cd src/FULL-W2V && make REGISTER;
	mv src/FULL-W2V/REGISTER ./REGISTER_$(HOSTNAME);
gdb_REGISTER:
	cd src/FULL-W2V && make gdb_REGISTER;
	mv src/FULL-W2V/gdb_REGISTER ./gdb_REGISTER_$(HOSTNAME);
expTable:
	cd src/FULL-W2V && make expTable;
expTable-no-cuda:
	cd src/FULL-W2V && make expTable-no-cuda;
# PWORD2VEC GROUP
pWord2Vec:
	cd src/pWord2Vec && make pWord2Vec;
	mv src/pWord2Vec/pWord2Vec ./pWord2Vec_$(HOSTNAME);
gdb_pWord2Vec:
	cd src/pWord2Vec && make gdb_pWord2Vec;
	mv src/pWord2Vec/gdb_pWord2Vec ./gdb_pWord2Vec_$(HOSTNAME);
pWord2Vec_mpi:
	cd src/pWord2Vec && make pWord2vec_mpi;
	mv src/pWord2Vec/pWord2Vec_mpi ./pWord2Vec_mpi_$(HOSTNAME);
# PSGNSCC GROUP
pSGNScc:
	cd src/pSGNScc && make pSGNScc;
	mv src/pSGNScc/pSGNScc ./pSGNScc_$(HOSTNAME);
gdb_pSGNScc:
	cd src/pSGNScc && make gdb_pSGNScc;
	mv src/pSGNScc/gdb_pSGNScc ./gdb_pSGNScc_$(HOSTNAME);
# ACCSGNS GROUP
accSGNS:
	cd src/accSGNS && make accSGNS;
	mv src/accSGNS/accSGNS ./accSGNS_$(HOSTNAME);
gdb_accSGNS:
	cd src/accSGNS && make gdb_accSGNS;
	mv src/accSGNS/gdb_accSGNS ./gdb_accSGNS_$(HOSTNAME);
# WOMBAT GROUP
wombat:
	cd src/wombat && make cuda;
	mv src/wombat/wombat ./wombat_$(HOSTNAME);
wombat_cpu:
	cd src/wombat && make intel;
	mv src/wombat/wombat_cpu ./wombat_cpu_$(HOSTNAME);

