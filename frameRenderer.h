#ifndef __FRAME_RENDERER_H__
#define __FRAME_RENDERER_H__

#include <cstdint>

struct Image;

class FrameRenderer {
public:
  virtual ~FrameRenderer(){};

  virtual const Image *getImage() = 0;

  virtual void setup() = 0;

  virtual void allocOutputImage() = 0;

  virtual void clearImage() = 0;

  virtual void advanceAnimation() = 0;

  virtual void render() = 0;
};

#endif
