
#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// #include "redEyeCPU.cpp"
// 
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

void radix_sort(unsigned int* const inputVals,
                unsigned int* const inputPos,
                unsigned int* const outputVals,
                unsigned int* const outputPos,
                const size_t numElems);

int main() {
  unsigned int *inputVals;
  unsigned int *inputPos;
  unsigned int *outputVals;
  unsigned int *outputPos;

  size_t numElems;

  std::string input_file = "red_eye_effect_5.jpg";
  std::string template_file = "red_eye_effect_template_5.jpg";
  std::string output_file = "fromGPU.jpg";
  std::string reference_file  = "fromCPU.jpg";

  //load the image and give us our input and output pointers
  preProcess(&inputVals, &inputPos, &outputVals, &outputPos, numElems, input_file, template_file);

  radix_sort(inputVals, inputPos, outputVals, outputPos, numElems);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  // int err = printf("GPU Code ran in: %f msecs.\n", timer.Elapsed());

  // if (err < 0) {
  //   //Couldn't print! Probably the student closed stdout - bad news
  //   std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
  //   exit(1);
  // }

  //check results and output the red-eye corrected image
  postProcess(outputVals, outputPos, numElems, output_file);

  // thrust::device_ptr<unsigned int> d_inputVals(inputVals);
  // thrust::device_ptr<unsigned int> d_inputPos(inputPos);

  // thrust::host_vector<unsigned int> h_inputVals(d_inputVals,
  //     d_inputVals + numElems);
  // thrust::host_vector<unsigned int> h_inputPos(d_inputPos,
  //     d_inputPos + numElems);

  // thrust::host_vector<unsigned int> h_outputVals(numElems);
  // thrust::host_vector<unsigned int> h_outputPos(numElems);

  // reference_calculation(&h_inputVals[0], &h_inputPos[0],
  //                       &h_outputVals[0], &h_outputPos[0],
  //                       numElems);

  // unsigned int *posPtr;
  // postProcess(valsPtr, posPtr, numElems, reference_file);

  // postProcess(&h_outputVals, &h_outputPos, numElems, reference_file);

  checkCudaErrors(cudaFree(inputVals));
  checkCudaErrors(cudaFree(inputPos));
  checkCudaErrors(cudaFree(outputVals));
  checkCudaErrors(cudaFree(outputPos));

  return 0;
}
