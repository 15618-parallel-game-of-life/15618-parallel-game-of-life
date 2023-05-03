import random
from cpu_renderer_wrapper import PyCpuRenderer

size = 5
pixel_size = 3

init = [1, 0, 1, 0, 1,
        0, 1, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0]

print(init[:3])

print(init)
init_frame = bytearray(init)

cpu_renderer = PyCpuRenderer(init_frame, size, pixel_size)

print(init_frame)

output = cpu_renderer.getFrame()
print(output)

cpu_renderer.advanceAnimation()

output = cpu_renderer.getFrame()
print(output)
