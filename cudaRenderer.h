#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#include <cstdio>
#include <cstring>
#include <utility>

#include "frameRenderer.h"
#include "image.h"

class CudaRenderer : public FrameRenderer
{
private:
  Image *image;
  uint8_t *initFrame;
  uint8_t *tmpFrame;
  int size, pixelSize;

  uint8_t *deviceCurrentFrame;
  uint8_t *deviceNextFrame;
  float *deviceImageData;

public:
  CudaRenderer(uint8_t *initFrame, int size, int pixelSize)
      : initFrame(initFrame), size(size), pixelSize(pixelSize)
  {
    image = nullptr;
    deviceCurrentFrame = nullptr;
    deviceNextFrame = nullptr;
    deviceImageData = nullptr;
    tmpFrame = new uint8_t[size * size];
  }

  virtual ~CudaRenderer();

  const uint8_t *getFrame();

  const Image *getImage();

  void setup();

  void allocOutputImage()
  {
    if (image)
      delete image;
    image = new Image(size * pixelSize, size * pixelSize);
  }

  void advanceAnimation();

  void render();
};

#endif
