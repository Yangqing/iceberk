// The fast pooling code implemented in C
// author: Yangqing Jia (jiayq@eecs.berkeley.edu)
// Copyright 2012

#include <cstring>
#include <cmath>

extern "C" {

void im2col(const double* imin,
            const int* imsize,
            const int* psize,
            const int stride,
            double* imout) {
    // The naive im2col implementation
    int ph = psize[0], pw = psize[1];
    int height = imsize[0], width = imsize[1], nchannels = imsize[2];
    double* current = imout;
    int segment = pw * nchannels;
    int endh = height - ph + 1;
    int endw = width - pw + 1;
    const double* src;
    for (int idxh = 0; idxh < endh; idxh += stride) {
        for (int idxw = 0; idxw < endw; idxw += stride) {
            // copy image[idxh:idxh+ph, idxw:idxw+pw, :]
            for (int i = idxh; i < idxh + ph; ++i) {
                // copy image[i, idxw:idxw+pw, :]
                src = imin + (i * width + idxw) * nchannels;
                for (int j = 0; j < segment; ++j) {
                    current[j] = src[j];
                }
                current += segment;
            }
        }
    }
} // im2col

} // extern "C"

