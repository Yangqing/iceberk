// The fast pooling code implemented in C
// author: Yangqing Jia (jiayq@eecs.berkeley.edu)
// Copyright 2012

#include <cstring>
#include <cmath>

#include <omp.h>

extern "C" {

void im2col(const double* imin,
            const int* imsize,
            const int* psize,
            const int stride,
            double* imout) {
    // The naive im2col implementation
    int ph = psize[0], pw = psize[1];
    int height = imsize[0], width = imsize[1], nchannels = imsize[2];
    int step_in = width * nchannels;
    int step_out = pw * nchannels;
    int height_out = (height - ph) / stride + 1;
    int width_out = (width - pw) / stride + 1;
#pragma omp parallel for
    for (int idxh = 0; idxh < height_out; ++idxh) {
        double* current = imout + idxh * width_out * ph * step_out;
        for (int idxw = 0; idxw < width_out; ++idxw) {
            // copy image[idxh:idxh+ph, idxw:idxw+pw, :]
            int hstart = idxh * stride;
            const double* src = imin + (hstart * width + idxw * stride) * nchannels;
            for (int i = hstart; i < hstart + ph; ++i) {
                // copy image[i, idxw:idxw+pw, :]
                for (int j = 0; j < step_out; ++j) {
                    current[j] = src[j];
                }
                current += step_out;
                src += step_in;
            }
        }
    }
} // im2col

} // extern "C"

