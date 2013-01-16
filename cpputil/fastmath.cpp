// The fast pooling code implemented in C
// author: Yangqing Jia (jiayq@eecs.berkeley.edu)
// Copyright 2012

#include <cstring>
#include <cmath>

# include <omp.h>

#define MAXPOOL 0
#define AVEPOOL 1
#define RMSPOOL 2

extern "C" {

int fastpooling(
        const double* const image, // Input image, [height*width*nchannels]
        const int height, 
        const int width,
        const int nchannels,
        const int gridh, // The grid size along the height
        const int gridw, // The grid size along the width
        const int method, // The pooling method
        double* output // output pooled features, 
                             // [gridh*gridw*nchannels]
        )
{
    int* counts = new int[gridh * gridw];
    memset(counts, 0, sizeof(int) * gridh * gridw);
    memset(output, 0, sizeof(double) * gridh * gridw * nchannels);
    
    switch (method) {
    case MAXPOOL:
        #pragma omp parallel for
        for (int k = 0; k < nchannels; ++k) {
            for (int i = 0; i < height; ++i) {
                int h_id = i * gridh / height;
                for (int j = 0; j < width; ++j) {
                    int w_id = j * gridw / width;
                    const double* image_hw = image + (i * width + j) * nchannels;
                    double* output_hw = output + (h_id * gridw + w_id) * nchannels;
                        output_hw[k] = (output_hw[k] > image_hw[k]) ? 
                                        output_hw[k] : image_hw[k];
                } // loop over width
            } // loop over height
        } // loop over channels
        break;
    case AVEPOOL:
        #pragma omp parallel for
        for (int i = 0; i < height; ++i) {
            int h_id = i * gridh / height;
            for (int j = 0; j < width; ++j) {
                int w_id = j * gridw / width;
                const double* image_hw = image + (i * width + j) * nchannels;
                double* output_hw = output + (h_id * gridw + w_id) * nchannels;
                #pragma omp atomic
                ++counts[h_id * gridw + w_id];
                for (int k = 0; k < nchannels; ++k) {
                    #pragma omp atomic
                    output_hw[k] += image_hw[k];
                } // loop over channels
            } // loop over width
        } // loop over height
        // Now, average
        #pragma omp parallel for
        for (int i = 0; i < gridh * gridw; ++i) {
            int count = counts[i];
            for (int k = 0; k < nchannels; ++k) {
                output[i * nchannels + k] /= count;
            }
        }
        break;
    case RMSPOOL:
        #pragma omp parallel for
        for (int i = 0; i < height; ++i) {
            int h_id = i * gridh / height;
            for (int j = 0; j < width; ++j) {
                int w_id = j * gridw / width;
                const double* image_hw = image + (i * width + j) * nchannels;
                double* output_hw = output + (h_id * gridw + w_id) * nchannels;
                #pragma omp atomic
                ++counts[h_id * gridw + w_id];
                for (int k = 0; k < nchannels; ++k) {
                    double sqvalue = image_hw[k] * image_hw[k];
                    #pragma omp atomic
                    output_hw[k] += sqvalue;
                } // loop over channels
            } // loop over width
        } // loop over height
        // Now, normalize
        #pragma omp parallel for
        for (int i = 0; i < gridh * gridw; ++i) {
            int count = counts[i];
            for (int k = 0; k < nchannels; ++k) {
                output[i * nchannels + k] = 
                    sqrt(output[i * nchannels + k] / count);
            }
        }
        break;
    default:
        delete[] counts;
        return 1;
    }
    delete[] counts;
    return 0;
}


int fast_oc_pooling(
        const double* const image, // Input image, [gridh*gridw*nchannels]
        const int gridh, // The grid size along the height
        const int gridw, // The grid size along the width
        const int nchannels,
        const int method, // The pooling method
        double* output // output pooled features, 
                             // [gridh*gridw*nchannels]
        )
{
    int num_rfs = gridh * (gridh + 1) * gridw * (gridw + 1) / 4;
    memset(output, 0, sizeof(double) * num_rfs * nchannels);
    const double* image_pq;
    double* output_pq = output;
    switch (method) {
    case MAXPOOL:
        for (int i = 0; i < gridh; ++i) {
            for (int j = i+1; j <= gridh; ++j) {
                for (int k = 0; k < gridw; ++k) {
                    for (int m = k+1; m <= gridw; ++m) {
                        // pool the cube [i,j) * [k,m)
                        for (int p = i; p < j; ++p) {
                            for (int q = k; q < m; ++q) {
                                image_pq = image + (p * gridw + q) * nchannels;
                                for (int s = 0; s < nchannels; ++s) {
                                    output_pq[s] = 
                                        (output_pq[s] > image_pq[s]) ? 
                                        output_pq[s] : image_pq[s];
                                }
                            }
                        }
                        output_pq += nchannels;
                    } // m: end of w
                } // k: start of w
            } // j: end of h
        } // i: start of h
        break;
    case AVEPOOL:
        for (int i = 0; i < gridh; ++i) {
            for (int j = i+1; j <= gridh; ++j) {
                for (int k = 0; k < gridw; ++k) {
                    for (int m = k+1; m <= gridw; ++m) {
                        // pool the cube [i,j) * [k,m)
                        for (int p = i; p < j; ++p) {
                            for (int q = k; q < m; ++q) {
                                image_pq = image + (p * gridw + q) * nchannels;
                                for (int s = 0; s < nchannels; ++s) {
                                    output_pq[s] += image_pq[s];
                                }
                            }
                        }
                        // Now, average
                        int rf_size = (j-i) * (m-k);
                        for (int s = 0; s < nchannels; ++s) {
                            output_pq[s] /= rf_size;
                        }
                        output_pq += nchannels;
                    } // m: end of w
                } // k: start of w
            } // j: end of h
        } // i: start of h
        break;
    case RMSPOOL: {
        int total_pixels = gridh * gridw * nchannels;
        double* image2 = new double[total_pixels];
        for (int i = 0; i < total_pixels; ++i) {
            image2[i] = image[i] * image[i];
        }
        for (int i = 0; i < gridh; ++i) {
            for (int j = i+1; j <= gridh; ++j) {
                for (int k = 0; k < gridw; ++k) {
                    for (int m = k+1; m <= gridw; ++m) {
                        // pool the cube [i,j) * [k,m)
                        for (int p = i; p < j; ++p) {
                            for (int q = k; q < m; ++q) {
                                image_pq = image2 + (p * gridw + q) * nchannels;
                                for (int s = 0; s < nchannels; ++s) {
                                    output_pq[s] += image_pq[s];
                                }
                            }
                        }
                        // Now, average
                        int rf_size = (j-i) * (m-k);
                        for (int s = 0; s < nchannels; ++s) {
                            output_pq[s] = sqrt(output_pq[s] / rf_size);
                        }
                        output_pq += nchannels;
                    } // m: end of w
                } // k: start of w
            } // j: end of h
        } // i: start of h
        delete[] image2;
        }break;
    default:
        return 1;
    }
    return 0;
}


} // extern "C"

