# import random
# from cpu_renderer_wrapper import PyCpuRenderer

# size = 5
# pixel_size = 3

# init = [1, 0, 1, 0, 1,
#         0, 1, 0, 0, 0,
#         0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0]

# print(init[:3])

# print(init)
# init_frame = bytearray(init)

# cpu_renderer = PyCpuRenderer(init_frame, size, pixel_size)

# print(init_frame)

# output = cpu_renderer.getFrame()
# print(output)

# cpu_renderer.advanceAnimation()

# output = cpu_renderer.getFrame()
# print(output)

# import multiprocessing


# def square(number):
#   return number * number


# if __name__ == '__main__':
#   numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#   # Create a Pool with the number of available CPU cores
#   with multiprocessing.Pool() as pool:
#     # Map the square function to the numbers list and collect the results
#     results = pool.map(square, numbers)

#   for number, squared in zip(numbers, results):
#     print(f"The square of {number} is {squared}")
import numpy as np
vector = np.random.normal(size=5)
