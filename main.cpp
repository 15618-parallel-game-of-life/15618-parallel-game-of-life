#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "cpuRenderer.h"
#include "cudaRenderer.h"
#include "cycleTimer.h"
#include "platformgl.h"

void startRendererWithDisplay(FrameRenderer *renderer);

void startBenchmark(FrameRenderer *renderer, int simCycles, int size,
                    const std::string &frameFilename);

int main(int argc, char *argv[]) {
  if (argc < 7) {
    std::cout << "Usage: " << argv[0]
              << " <size> <input_file> <enable_gpu> <sim cycles> <pixel_size> "
                 "<output_file>\n";
    return 1;
  }
  int frameSize = std::atoi(argv[1]);
  std::ifstream in(argv[2]);
  if (!in) {
    std::cout << "Error opening input file argv[2]\n";
    return 1;
  }
  int enableGpu = std::atoi(argv[3]);
  int simCycles = std::atoi(argv[4]);
  int pixelSize = std::atoi(argv[5]);
  std::string frameFilename = argv[6];

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

  FrameRenderer *renderer;
  if (enableGpu)
    renderer = new CudaRenderer(initFrame, frameSize, pixelSize);
  else
    renderer = new CpuRenderer(initFrame, frameSize, pixelSize);

  renderer->allocOutputImage();
  renderer->setup();

  {
    const uint8_t *currentFrame = renderer->getFrame();
    char filename[1024];
    sprintf(filename, "outputs/%s_1111.txt", frameFilename.c_str());
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
      fprintf(stderr, "Error: could not open %s for write\n", filename);
      exit(1);
    }
    for (int i = 0; i < frameSize; i++) {
      for (int j = 0; j < frameSize; j++) {
        fprintf(fp, "%d ", currentFrame[i * frameSize + j]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }

  if (simCycles >= 0) {
    startBenchmark(renderer, simCycles, frameSize, frameFilename);
  } else {
    glutInit(&argc, argv);
    startRendererWithDisplay(renderer);
  }

  delete[] initFrame;

  return 0;
}

void startBenchmark(FrameRenderer *renderer, int simCycles, int size,
                    const std::string &frameFilename) {

  double totalAdvanceTime = 0.f;
  double totalRenderTime = 0.f;
  double totalFileSaveTime = 0.f;
  double totalTime = 0.f;
  double startTime = 0.f;

  printf("\nRunning benchmark, %d frames ...\n", simCycles);
  startTime = CycleTimer::currentSeconds();

  for (int frame = 0; frame < simCycles; frame++) {
    double startAdvanceTime = CycleTimer::currentSeconds();

    renderer->advanceAnimation();

    double endAdvanceTime = CycleTimer::currentSeconds();

    renderer->render();

    double endRenderTime = CycleTimer::currentSeconds();

    const uint8_t *currentFrame = renderer->getFrame();

    char filename[1024];
    sprintf(filename, "outputs/%s_%04d.txt", frameFilename.c_str(), frame);
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
      fprintf(stderr, "Error: could not open %s for write\n", filename);
      exit(1);
    }
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        fprintf(fp, "%d ", currentFrame[i * size + j]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
    double endFileSaveTime = CycleTimer::currentSeconds();

    totalAdvanceTime += endAdvanceTime - startAdvanceTime;
    totalRenderTime += endRenderTime - endAdvanceTime;
    totalFileSaveTime += endFileSaveTime - endRenderTime;
  }
  double endTime = CycleTimer::currentSeconds();
  totalTime = endTime - startTime;

  printf("Advance:  %.4f ms\n", 1000.f * totalAdvanceTime / simCycles);
  printf("Render:   %.4f ms\n", 1000.f * totalRenderTime / simCycles);
  printf("Total:    %.4f ms\n",
         1000.f * (totalAdvanceTime + totalRenderTime) / simCycles);
  printf("File IO:  %.4f ms\n", 1000.f * totalFileSaveTime / simCycles);
  printf("\n");
  printf("Overall:  %.4f sec (note units are seconds)\n", totalTime);
}
