CPPFLAGS = -I../../include -I../include

OBJS = auxiliary_eigs.o \
	convergence.o \
	correction.o \
	factorize.o \
	globalsum.o \
	init.o \
	inner_solve.o \
	main_iter.o \
	ortho.o \
	primme.o \
	primme_f77.o \
	primme_interface.o \
	restart.o \
	solve_projection.o \
	update_W.o \
	update_projection.o

.PHONY: all clean

all: $(OBJS)

.c.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) -DF77UNDERSCORE -DUSE_DOUBLE -c $< -o $@

clean:
	$(RM) *.o
