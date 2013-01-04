CC = g++
CCFLAGS = -fPIC -O3 -Wall -pedantic -ansi -ffast-math -msse -msse2
LINKFLAGS = -shared -Wl
INPUT = fastmath.cpp im2col.cpp clip.cpp
TARGET = libcpputil.so
all:
	$(CC) -c $(CCFLAGS) $(INPUT)
	$(CC) $(LINKFLAGS) -o $(TARGET) *.o
clean:
	rm *.so
	rm *.o
