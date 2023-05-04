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
  bool play;

public:
  CpuRenderer(uint8_t *initFrame, int size, int pixelSize, bool play)
      : initFrame(initFrame), size(size), pixelSize(pixelSize), play(play) {
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

  const uint8_t *getFrame() { return currentFrame; }

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
        if (play) {
          int count1 = 0, count2 = 0;
          for (int ii = i - 1; ii <= i + 1; ii++) {
            for (int jj = j - 1; jj <= j + 1; jj++) {
              int idx2 = ((ii + size) % size) * size + (jj + size) % size;
              if (idx2 == idx)
                continue;
              if (currentFrame[idx2] == 1)
                count1++;
              if (currentFrame[idx2] == 2)
                count2++;
            }
          }
          if (count1 + count2 >= 4 || count1 + count2 <= 1)
            nextFrame[idx] = 0;
          else if (count1 + count2 == 3)
            nextFrame[idx] = (count1 > count2) ? 1 : 2;
        } else {
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
            if (play) {
              imgPtr[0] = (c == 1) ? 1.0f : 0.0f;
              imgPtr[1] = 0.0f;
              imgPtr[2] = (c == 2) ? 1.0f : 0.0f;
            } else {
              imgPtr[0] = (c == 1) ? 1.0f : 0.0f;
              imgPtr[1] = (c == 1) ? 1.0f : 0.0f;
              imgPtr[2] = (c == 1) ? 1.0f : 0.0f;
            }
            imgPtr[3] = 1.0f;
          }
        }
      }
    }
  }
};

#endif
