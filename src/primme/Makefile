OBJS = eigs/auxiliary_eigs.o \
	eigs/convergence.o \
	eigs/correction.o \
	eigs/factorize.o \
	eigs/globalsum.o \
	eigs/init.o \
	eigs/inner_solve.o \
	eigs/main_iter.o \
	eigs/ortho.o \
	eigs/primme.o \
	eigs/primme_f77.o \
	eigs/primme_interface.o \
	eigs/restart.o \
	eigs/solve_projection.o \
	eigs/update_W.o \
	eigs/update_projection.o \
	linalg/auxiliary.o \
	linalg/blaslapack.o \
	linalg/wtime.o

.PHONY: all clean

all: libprimme.a

$(OBJS):
	cd eigs && $(MAKE) all CC="$(CC)" CFLAGS="$(CFLAGS)"
	cd linalg && $(MAKE) all CC="$(CC)" CFLAGS="$(CFLAGS)"

libprimme.a: $(OBJS)
	@$(RM) $@
	$(AR) rcs $@ $(OBJS)

clean:
	$(RM) *.a
	cd eigs && $(MAKE) clean
	cd linalg && $(MAKE) clean
