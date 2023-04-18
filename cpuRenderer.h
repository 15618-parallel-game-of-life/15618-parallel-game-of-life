#ifndef __CPU_RENDERER_H__
#define __CPU_RENDERER_H__

#include <cstdio>
#include <cstring>
#include <utility>

#include "frameRenderer.h"
#include "image.h"

class CpuRenderer : public FrameRenderer {
private:
  Image *image;
  uint8_t *initFrame;
  uint8_t *currentFrame;
  uint8_t *nextFrame;
  int size, pixelSize;

public:
  CpuRenderer(uint8_t *initFrame, int size, int pixelSize)
      : initFrame(initFrame), size(size), pixelSize(pixelSize) {
    image = nullptr;
    currentFrame = new uint8_t[size * size];
    nextFrame = new uint8_t[size * size];
    memcpy(currentFrame, initFrame, size * size * sizeof(uint8_t));
  }

  virtual ~CpuRenderer() {
    if (image)
      delete image;
    delete[] currentFrame;
    delete[] nextFrame;
  }

  const Image *getImage() { return image; }

  void setup() {}

  void allocOutputImage() {
    if (image)
      delete image;
    image = new Image(size * pixelSize, size * pixelSize);
  }

  void advanceAnimation() {
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        int idx = i * size + j;
        int count = 0;
        for (int ii = i - 1; ii <= i + 1; ii++) {
          for (int jj = j - 1; jj <= j + 1; jj++) {
            int idx2 = ((ii + size) % size) * size + (jj + size) % size;
            if (idx2 == idx)
              continue;
            if (currentFrame[idx2])
              count++;
          }
        }
        if (count == 3 || (count == 2 && currentFrame[idx]))
          nextFrame[idx] = 1;
        else
          nextFrame[idx] = 0;
      }
    }
    std::swap(currentFrame, nextFrame);
  }

  void render() {
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        int idx = i * size + j;
        int c = currentFrame[idx];
        for (int ii = 0; ii < pixelSize; ii++) {
          for (int jj = 0; jj < pixelSize; jj++) {
            int idx2 =
                (i * pixelSize + ii) * size * pixelSize + (j * pixelSize + jj);

            float *imgPtr = &image->data[4 * idx2];
            imgPtr[0] = (c == 1) ? 1.0f : 0.0f;
            imgPtr[1] = (c == 1) ? 1.0f : 0.0f;
            imgPtr[2] = (c == 1) ? 1.0f : 0.0f;
            imgPtr[3] = 1.0f;
          }
        }
      }
    }
  }
};

#endif
