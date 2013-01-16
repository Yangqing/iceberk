CC = g++
CCFLAGS = -fPIC -O3 -Wall -pedantic -ansi -ffast-math -msse -msse2 -fopenmp
LINKFLAGS = -shared -Wl -lgomp
INPUT = fastpool.cpp im2col.cpp
TARGET = libcpputil.so
all:
	$(CC) -c $(CCFLAGS) $(INPUT)
	$(CC) $(LINKFLAGS) -o $(TARGET) *.o
clean:
	rm *.so
	rm *.o
