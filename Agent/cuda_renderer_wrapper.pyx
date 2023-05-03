cdef extern from "../CudaRenderer.h":
    cdef cppclass CudaRenderer:
        CudaRenderer(char *initFrame, int size, int pixelSize)
        # Add other public member function declarations here if necessary

cdef class PyCudaRenderer:
    cdef CudaRenderer *thisptr

    def __cinit__(self, char *initFrame, int size, int pixelSize):
        self.thisptr = new CudaRenderer(initFrame, size, pixelSize)

    def __dealloc__(self):
        del self.thisptr

    # Add Python wrapper functions for other public member functions if necessary