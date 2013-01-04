// The clip code implemented in C
// author: Yangqing Jia (jiayq@eecs.berkeley.edu)
// Copyright 2012

#include <cstring>
#include <cmath>
#include <algorithm>

using namespace std;

#define CLIP_LOWER 1
#define CLIP_UPPER 2

extern "C" {

void clip(double* array,
          const int size,
          const double lb,
          const double ub,
          const int mode) {
    if (mode & CLIP_LOWER) {
        for (int i = 0; i < size; ++i) {
            array[i] = min(array[i], lb); 
        }
    }
    if (mode & CLIP_UPPER) {
        for (int i = 0; i < size; ++i) {
            array[i] = max(array[i], ub); 
        }
    }
}

} // extern "C"

