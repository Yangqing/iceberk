CC = g++
CCFLAGS = -fPIC -O3 -Wall -pedantic -ansi -ffast-math -msse -msse2
LINKFLAGS = -shared -Wl
INPUT = fastmath.cpp
TARGET = libfastmath.so
all:
	$(CC) -c $(CCFLAGS) $(INPUT)
	$(CC) $(LINKFLAGS) -o $(TARGET) *.o
clean:
	rm *.so
	rm *.o
