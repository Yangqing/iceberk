// The fast pooling code implemented in C
// author: Yangqing Jia (jiayq@eecs.berkeley.edu)
// Copyright 2012

#include <cstring>
#include <cmath>

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
        double* const output // output pooled features, 
                             // [gridh*gridw*nchannels]
        )
{
    int* counts = new int[gridh * gridw];
    memset(counts, 0, sizeof(int) * gridh * gridw);
    memset(output, 0, sizeof(double) * gridh * gridw * nchannels);
    
    switch (method) {
    case MAXPOOL:
        for (int i = 0; i < height; ++i) {
            int h_id = i * gridh / height;
            for (int j = 0; j < width; ++j) {
                int w_id = j * gridw / width;
                const double* image_hw = image + (i * width + j) * nchannels;
                double* output_hw = output + (h_id * gridw + w_id) * nchannels;
                for (int k = 0; k < nchannels; ++k) {
                    output_hw[k] = (output_hw[k] > image_hw[k]) ? 
                                    output_hw[k] : image_hw[k];
                } // loop over channels
            } // loop over width
        } // loop over height
        break;
    case AVEPOOL:
        for (int i = 0; i < height; ++i) {
            int h_id = i * gridh / height;
            for (int j = 0; j < width; ++j) {
                int w_id = j * gridw / width;
                const double* image_hw = image + (i * width + j) * nchannels;
                double* output_hw = output + (h_id * gridw + w_id) * nchannels;
                ++counts[h_id * gridw + w_id];
                for (int k = 0; k < nchannels; ++k) {
                    output_hw[k] += image_hw[k];
                } // loop over channels
            } // loop over width
        } // loop over height
        // Now, average
        for (int i = 0; i < gridh * gridw; ++i) {
            int count = counts[i];
            for (int k = 0; k < nchannels; ++k) {
                output[i * nchannels + k] /= count;
            }
        }
        break;
    case RMSPOOL:
        for (int i = 0; i < height; ++i) {
            int h_id = i * gridh / height;
            for (int j = 0; j < width; ++j) {
                int w_id = j * gridw / width;
                const double* image_hw = image + (i * width + j) * nchannels;
                double* output_hw = output + (h_id * gridw + w_id) * nchannels;
                ++counts[h_id * gridw + w_id];
                for (int k = 0; k < nchannels; ++k) {
                    output_hw[k] += image_hw[k] * image_hw[k];
                } // loop over channels
            } // loop over width
        } // loop over height
        // Now, normalize
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

void faststd(const double* const data, // input data
             const double* const mean, // input mean
             const int nrows, // num of rows
             const int ncols, // num of cols
             const int axis, // axis along which to do std
             double* const std // output std
             )
{
    int num_data;
    int num_output;
    double datum;
    double* comp = NULL;
    if (axis == 0) {
        num_data = nrows;
        num_output = ncols;
        comp = new double[num_output];
        memset(comp, 0, sizeof(double) * num_output);
        memset(std, 0, sizeof(double) * num_output);
        for (int i = 0; i < nrows; ++i) {
            for (int j = 0; j < ncols; ++j) {
                datum = data[i*ncols+j] - mean[j];
                std[j] += datum * datum;
                comp[j] += datum;
            }
        }
    } else {
        num_data = ncols;
        num_output = nrows;
        comp = new double[num_output];
        memset(comp, 0, sizeof(double) * num_output);
        memset(std, 0, sizeof(double) * num_output);
        for (int i = 0; i < nrows; ++i) {
            for (int j = 0; j < ncols; ++j) {
                datum = data[i*ncols+j] - mean[i];
                std[i] += datum * datum;
                comp[i] += datum;
            }
        }
    }
    // now, do normalization
    for (int i = 0; i < num_output; ++i) {
        //std[i] = sqrt(std[i] / num_data);
        std[i] = sqrt((std[i] - comp[i]*comp[i] / num_data) / num_data);
    }
    delete[] comp;
}

} // extern "C"

