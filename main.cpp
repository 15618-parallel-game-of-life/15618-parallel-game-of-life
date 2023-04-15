#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "cpuRenderer.h"
#include "platformgl.h"

void startRendererWithDisplay(FrameRenderer *renderer);

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cout << "Usage: " << argv[0]
              << " <size> <input_file> <sim cycles> <pixel_size>\n";
    return 1;
  }
  int frameSize = std::atoi(argv[1]);
  std::ifstream in(argv[2]);
  if (!in) {
    std::cout << "Error opening input file argv[2]\n";
    return 1;
  }
  int simCycles = std::atoi(argv[3]);
  int pixelSize = std::atoi(argv[4]);

  auto *initFrame = new uint8_t[frameSize * frameSize];

  for (int i = 0; i < frameSize; i++) {
    for (int j = 0; j < frameSize; j++) {
      char c;
      in >> c;
      initFrame[i * frameSize + j] = (c == '0') ? 0 : 1;
      if (c != '0' && c != '1') {
        std::cout << "Error in input file, location (" << i << ", " << j
                  << "), c = " << c << "\n";
        return 1;
      }
    }
  }

  auto *renderer = new CpuRenderer(initFrame, frameSize, pixelSize);

  renderer->allocOutputImage();
  renderer->setup();

  if (simCycles >= 0) {
    // startBenchmark(renderer, benchmarkFrameStart,
    //                benchmarkFrameEnd - benchmarkFrameStart, frameFilename);
  } else {
    glutInit(&argc, argv);
    startRendererWithDisplay(renderer);
  }

  delete[] initFrame;

  return 0;
}
