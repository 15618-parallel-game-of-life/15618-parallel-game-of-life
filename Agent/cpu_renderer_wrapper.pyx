from libc.stdint cimport uint8_t

cdef extern from "../cpuRenderer.h":
    cdef cppclass CpuRenderer:
        CpuRenderer(uint8_t *initFrame, int size, int pixelSize)
        void advanceAnimation()
        uint8_t *getFrame()

cdef class PyCpuRenderer:
    cdef CpuRenderer *thisptr

    def __cinit__(self, uint8_t *initFrame, int size, int pixelSize):
        self.thisptr = new CpuRenderer(initFrame, size, pixelSize)

    def __dealloc__(self):
        del self.thisptr
    
    def advanceAnimation(self):
        return self.thisptr.advanceAnimation()
    
    def getFrame(self):
        return self.thisptr.getFrame()