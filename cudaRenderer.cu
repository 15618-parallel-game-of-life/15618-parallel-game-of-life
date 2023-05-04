#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"

#define SQRT_NUM_THREADS 16

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n",
            cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}
#else
#define cudaCheckError(ans) ans
#endif

// data structure definition start here

// global constants definition
struct GlobalConstants
{
  int size;
  int pixelSize;
  float *imageData;
};

__constant__ GlobalConstants cuConstRendererParams;

// data structure definition end here

// CUDA kernel functions start here

__global__ void kernelAdvanceAnimationWithBuffer(uint8_t *current, uint8_t *next)
{
  __shared__ uint8_t buffer[SQRT_NUM_THREADS + 2][SQRT_NUM_THREADS + 2];

  int idxX = blockIdx.x * blockDim.x + threadIdx.x;
  int idxY = blockIdx.y * blockDim.y + threadIdx.y;

  int size = cuConstRendererParams.size;

  // load the neighborhood elements into local buffer
  int bufferX = threadIdx.x + 1;
  int bufferY = threadIdx.y + 1;
  int globalIdx = idxX * size + idxY;

  if (idxX < size && idxY < size)
  {
    buffer[bufferX][bufferY] = current[globalIdx];

    // boundaries
    if (threadIdx.x == 0)
      buffer[0][bufferY] = current[((idxX - 1 + size) % size) * size + idxY];
    if (threadIdx.x == blockDim.x - 1)
      buffer[SQRT_NUM_THREADS + 1][bufferY] = current[((idxX + 1) % size) * size + idxY];
    if (threadIdx.y == 0)
      buffer[bufferX][0] = current[idxX * size + ((idxY - 1 + size) % size)];
    if (threadIdx.y == blockDim.y - 1)
      buffer[bufferX][SQRT_NUM_THREADS + 1] = current[idxX * size + ((idxY + 1) % size)];
    if (threadIdx.x == 0 && threadIdx.y == 0)
      buffer[0][0] = current[((idxX - 1 + size) % size) * size + ((idxY - 1 + size) % size)];
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
      buffer[0][SQRT_NUM_THREADS + 1] = current[((idxX - 1 + size) % size) * size + ((idxY + 1) % size)];
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
      buffer[SQRT_NUM_THREADS + 1][0] = current[((idxX + 1) % size) * size + ((idxY - 1 + size) % size)];
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
      buffer[SQRT_NUM_THREADS + 1][SQRT_NUM_THREADS + 1] = current[((idxX + 1) % size) * size + ((idxY + 1) % size)];
  }

  __syncthreads();

  if (idxX >= size || idxY >= size)
    return;

  int count = 0;
  for (int ii = -1; ii <= 1; ii++)
  {
    for (int jj = -1; jj <= 1; jj++)
    {
      if (buffer[bufferX + ii][bufferY + jj] && !(ii == 0 && jj == 0))
        count++;
    }
  }

  // update next state
  if (count == 3 || (count == 2 && buffer[bufferX][bufferY]))
    next[globalIdx] = 1;
  else
    next[globalIdx] = 0;
}

__global__ void kernelAdvanceAnimation(uint8_t *current, uint8_t *next)
{
  int idxX = blockIdx.x * blockDim.x + threadIdx.x;
  int idxY = blockIdx.y * blockDim.y + threadIdx.y;

  int size = cuConstRendererParams.size;

  if (idxX >= size || idxY >= size)
    return;

  int idx = (idxX * size + idxY);
  int count = 0;
  for (int ii = idxX - 1; ii <= idxX + 1; ii++)
  {
    for (int jj = idxY - 1; jj <= idxY + 1; jj++)
    {
      int idx2 = ((ii + size) % size) * size + (jj + size) % size;
      if (idx2 == idx)
        continue;
      if (current[idx2])
        count++;
    }
  }

  if (count == 3 || (count == 2 && current[idx]))
    next[idx] = 1;
  else
    next[idx] = 0;
}

__global__ void kernelRenderFrame(uint8_t *frame)
{
  int imageX = blockIdx.x * blockDim.x + threadIdx.x;
  int imageY = blockIdx.y * blockDim.y + threadIdx.y;

  int size = cuConstRendererParams.size;
  int pixelSize = cuConstRendererParams.pixelSize;

  if (imageX >= (size * pixelSize) || imageY >= (size * pixelSize))
    return;

  int idxX = imageX / pixelSize;
  int idxY = imageY / pixelSize;

  int idx = (idxX * size + idxY);
  uint8_t c = frame[idx];
  float4 imgPixel;
  if (c == 1)
    imgPixel = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  else
    imgPixel = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

  int idx4 = 4 * (imageX * size * pixelSize + imageY);
  *(float4 *)(&cuConstRendererParams.imageData[idx4]) = imgPixel;
}

// CUDA kernel functions end here

// C++ class member functions start here

CudaRenderer::~CudaRenderer()
{
  if (image)
    delete image;
  if (deviceCurrentFrame)
  {
    cudaFree(deviceCurrentFrame);
    cudaFree(deviceNextFrame);
    cudaFree(deviceImageData);
  }
  if (tmpFrame)
  {
    delete[] tmpFrame;
  }
}

const uint8_t *CudaRenderer::getFrame()
{
  cudaMemcpy(tmpFrame,
             deviceCurrentFrame,
             sizeof(uint8_t) * size * size,
             cudaMemcpyDeviceToHost);

  return tmpFrame;
}

const Image *CudaRenderer::getImage()
{
  cudaMemcpy(image->data,
             deviceImageData,
             sizeof(float) * 4 * image->width * image->height,
             cudaMemcpyDeviceToHost);

  return image;
}

void CudaRenderer::setup()
{
  int deviceCount = 0;
  std::string name;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  // printf("---------------------------------------------------------\n");
  // printf("Initializing CUDA for CudaRenderer\n");
  // printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++)
  {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    name = deviceProps.name;

    // printf("Device %d: %s\n", i, deviceProps.name);
    // printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    // printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    // printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  // printf("---------------------------------------------------------\n");

  // allocate memory storage on device
  cudaMalloc(&deviceCurrentFrame, sizeof(uint8_t) * size * size);
  cudaMalloc(&deviceNextFrame, sizeof(uint8_t) * size * size);
  cudaMalloc(&deviceImageData, sizeof(float) * 4 * size * size * pixelSize * pixelSize);

  // copy data from input to device memory
  cudaMemcpy(deviceCurrentFrame, initFrame, sizeof(uint8_t) * size * size, cudaMemcpyHostToDevice);

  // initialize global constants with struct
  GlobalConstants params;
  params.size = size;
  params.pixelSize = pixelSize;
  params.imageData = deviceImageData;

  cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));
}

void CudaRenderer::advanceAnimation()
{
  dim3 blockDim(SQRT_NUM_THREADS, SQRT_NUM_THREADS, 1);
  dim3 gridDim(
      (image->width + blockDim.x - 1) / blockDim.x,
      (image->height + blockDim.y - 1) / blockDim.y);

  // pixel-level parallelism
  kernelAdvanceAnimationWithBuffer<<<gridDim, blockDim>>>(deviceCurrentFrame, deviceNextFrame);
  // kernelAdvanceAnimation<<<gridDim, blockDim>>>(deviceCurrentFrame, deviceNextFrame);
  cudaCheckError(cudaDeviceSynchronize());

  // swap currentFrame and nextFrame
  std::swap(deviceCurrentFrame, deviceNextFrame);
}

void CudaRenderer::render()
{
  dim3 blockDim(SQRT_NUM_THREADS, SQRT_NUM_THREADS, 1);
  dim3 gridDim(
      (image->width + blockDim.x - 1) / blockDim.x,
      (image->height + blockDim.y - 1) / blockDim.y);

  // pixel-level parallelism
  kernelRenderFrame<<<gridDim, blockDim>>>(deviceCurrentFrame);
  cudaCheckError(cudaDeviceSynchronize());
}

// C++ class member functions end here
