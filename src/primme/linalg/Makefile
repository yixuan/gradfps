CPPFLAGS = -I../../include -I../include

OBJS = auxiliary.o \
	blaslapack.o \
	wtime.o

.PHONY: all clean

all: $(OBJS)

.c.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) -DF77UNDERSCORE -DUSE_DOUBLE -c $< -o $@

clean:
	$(RM) *.o
