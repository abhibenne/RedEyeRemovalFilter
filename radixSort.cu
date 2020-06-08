#include <float.h>
#include <math.h>
#include <stdio.h>

#include "utils.h"

__global__
void histogram_kernel(unsigned int pass,
                      unsigned int * d_bins,
                      unsigned int* const d_input,
                      const int size) {
  int mid = threadIdx.x + blockDim.x * blockIdx.x;
  if (mid >= size)
    return;
  unsigned int one = 1;
  int bin = ((d_input[mid] & (one << pass)) == (one << pass)) ? 1 : 0;
  if (bin)
    atomicAdd(&d_bins[1], 1);
  else
    atomicAdd(&d_bins[0], 1);
}

// we will run 1 exclusive scan, but then when we
// do the move, for zero vals, we iwll take mid - val of scan there
__global__
void exclusive_scan_kernel(unsigned int pass,
                           unsigned int const * d_inputVals,
                           unsigned int * d_output,
                           const int size,
                           unsigned int base,
                           unsigned int threadSize) {
  int mid = threadIdx.x + threadSize * base;
  int block = threadSize * base;
  unsigned int one = 1;

  if (mid >= size)
    return;

  unsigned int val = 0;
  if (mid > 0)
    val = ((d_inputVals[mid - 1] & (one << pass))  == (one << pass)) ? 1 : 0;
  else
    val = 0;

  d_output[mid] = val;

  __syncthreads();

  for (int s = 1; s <= threadSize; s *= 2) {
    int spot = mid - s;

    if (spot >= 0 && spot >=  threadSize * base)
      val = d_output[spot];
    __syncthreads();
    if (spot >= 0 && spot >= threadSize * base)
      d_output[mid] += val;
    __syncthreads();
  }
  if (base > 0)
    d_output[mid] += d_output[base * threadSize - 1];
}

__global__
void move_kernel(
  unsigned int pass,
  unsigned int* const d_inputVals,
  unsigned int* const d_inputPos,
  unsigned int* d_outputVals,
  unsigned int* d_outputPos,
  unsigned int* d_outputMove,
  unsigned int* const d_scanned,
  unsigned int  one_pos,
  const size_t numElems) {

  int mid = threadIdx.x + blockDim.x * blockIdx.x;
  if (mid >= numElems)
    return;

  unsigned int scan = 0;
  unsigned int base = 0;
  unsigned int one = 1;
  if ( ( d_inputVals[mid] & (one << pass)) == (1 << pass)) {
    scan = d_scanned[mid];
    base = one_pos;
  } else {
    scan = (mid) - d_scanned[mid];
    base = 0;
  }

  d_outputMove[mid] = base + scan;
  d_outputPos[base + scan]  = d_inputPos[mid]; //d_inputPos[0];
  d_outputVals[base + scan] = d_inputVals[mid]; //base+scan;//d_inputVals[0];

}

// max size for n/d better one
int get_max_size(int n, int d) {
  return (int)ceil( (float)n / (float)d ) + 1;
}

// host function for radix sort
void radix_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  unsigned int* d_bins;
  unsigned int  h_bins[2];
  unsigned int* d_scanned;
  unsigned int* d_moved;
  const size_t histo_size = 2 * sizeof(unsigned int);
  const size_t arr_size   = numElems * sizeof(unsigned int);

  checkCudaErrors(cudaMalloc(&d_bins, histo_size));
  checkCudaErrors(cudaMalloc(&d_scanned, arr_size));
  checkCudaErrors(cudaMalloc(&d_moved, arr_size));


  // for histogram kernel defined here
  dim3 thread_dim(1024);
  dim3 hist_block_dim(get_max_size(numElems, thread_dim.x));


  for (unsigned int pass = 0; pass < 32; pass++) {
    unsigned int one = 1;
    checkCudaErrors(cudaMemset(d_bins, 0, histo_size));
    checkCudaErrors(cudaMemset(d_scanned, 0, arr_size));
    checkCudaErrors(cudaMemset(d_outputVals, 0, arr_size));
    checkCudaErrors(cudaMemset(d_outputPos, 0, arr_size));

    histogram_kernel <<< hist_block_dim, thread_dim>>>(pass, d_bins, d_inputVals, numElems);
    cudaDeviceSynchronize(); 
    // checkCudaErrors(cudaGetLastError());

    cudaMemcpy(&h_bins, d_bins, histo_size, cudaMemcpyDeviceToHost);

    // printf("debugging %d %d %d %d %d \n", h_bins[0], h_bins[1], h_bins[0] + h_bins[1], numElems, (one << pass));

    for (int i = 0; i < get_max_size(numElems, thread_dim.x); i++) {
      exclusive_scan_kernel <<< dim3(1), thread_dim>>>(pass,d_inputVals,d_scanned,numElems,i,thread_dim.x);
      cudaDeviceSynchronize(); 
      // checkCudaErrors(cudaGetLastError());
    }
    // calculate the move positions
    move_kernel <<< hist_block_dim, thread_dim>>>(
      pass,
      d_inputVals,
      d_inputPos,
      d_outputVals,
      d_outputPos,
      d_moved,
      d_scanned,
      h_bins[0],
      numElems
    );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, arr_size, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, arr_size, cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }
  checkCudaErrors(cudaFree(d_moved));
  checkCudaErrors(cudaFree(d_scanned));
  checkCudaErrors(cudaFree(d_bins));
}