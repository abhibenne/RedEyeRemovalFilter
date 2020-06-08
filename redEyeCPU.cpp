
#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <algorithm>
// For memset
#include <cstring>

void reference_calculation(unsigned int* inputVals,
                           unsigned int* inputPos,
                           unsigned int* outputVals,
                           unsigned int* outputPos,
                           const size_t numElems)
{
  const int numBits = 1;
  const int numBins = 1 << numBits;

  unsigned int *binHistogram = new unsigned int[numBins];
  unsigned int *binScan      = new unsigned int[numBins];

  unsigned int *vals_src = inputVals;
  unsigned int *pos_src  = inputPos;

  unsigned int *vals_dst = outputVals;
  unsigned int *pos_dst  = outputPos;

  //a simple radix sort - only guaranteed to work for numBits that are multiples of 2
  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
    unsigned int mask = (numBins - 1) << i;

    memset(binHistogram, 0, sizeof(unsigned int) * numBins); //zero out the bins
    memset(binScan, 0, sizeof(unsigned int) * numBins); //zero out the bins

    //perform histogram of data & mask into bins
    for (unsigned int j = 0; j < numElems; ++j) {
      unsigned int bin = (vals_src[j] & mask) >> i;
      binHistogram[bin]++;
    }

    //perform exclusive prefix sum (scan) on binHistogram to get starting
    //location for each bin
    for (unsigned int j = 1; j < numBins; ++j) {
      binScan[j] = binScan[j - 1] + binHistogram[j - 1];
    }

    //Gather everything into the correct location
    //need to move vals and positions
    for (unsigned int j = 0; j < numElems; ++j) {
      unsigned int bin = (vals_src[j] & mask) >> i;
      vals_dst[binScan[bin]] = vals_src[j];
      pos_dst[binScan[bin]]  = pos_src[j];
      binScan[bin]++;
    }

    //swap the buffers (pointers only)
    std::swap(vals_dst, vals_src);
    std::swap(pos_dst, pos_src);
  }

  //we did an even number of iterations, need to copy from input buffer into output
  std::copy(inputVals, inputVals + numElems, outputVals);
  std::copy(inputPos, inputPos + numElems, outputPos);

  delete[] binHistogram;
  delete[] binScan;
}


void preProcess(unsigned int **inputVals,
                unsigned int **inputPos,
                unsigned int **outputVals,
                unsigned int **outputPos,
                size_t &numElems,
                const std::string& filename,
                const std::string& template_file);

void postProcess(const unsigned int* const outputVals,
                 const unsigned int* const outputPos,
                 const size_t numElems,
                 const std::string& output_file);

int main() {
  unsigned int *inputVals;
  unsigned int *inputPos;
  unsigned int *outputVals;
  unsigned int *outputPos;

  size_t numElems;

  std::string input_file = "red_eye_effect_5.jpg";
  std::string template_file = "red_eye_effect_template_5.jpg";
  std::string output_file = "outputCPU.jpg";


  preProcess(&inputVals, &inputPos, &outputVals, &outputPos, numElems, input_file, template_file);

  thrust::host_vector<unsigned int> h_inputVals(inputVals);
  thrust::host_vector<unsigned int> h_inputPos(inputPos);

  thrust::host_vector<unsigned int> h_outputVals(numElems);
  thrust::host_vector<unsigned int> h_outputPos(numElems);

  reference_calculation(&h_inputVals[0], &h_inputPos[0],
                        &h_outputVals[0], &h_outputPos[0],
                        numElems);

  postProcess(h_outputVals, h_outputPos, numElems, output_file);
  return 0;
}
