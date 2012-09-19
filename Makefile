CC = g++
CCFLAGS = -fPIC -O3 -Wall -pedantic -ansi -ffast-math -msse -msse2
LINKFLAGS = -shared -Wl
TARGET = libfastpool.so
all:
	$(CC) -c $(CCFLAGS) fastpool.cpp
	$(CC) $(LINKFLAGS) -o $(TARGET) *.o
clean:
	rm *.so
	rm *.o
