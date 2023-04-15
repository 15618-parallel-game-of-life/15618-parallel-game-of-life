import random
import sys

size = 100
mode = 'random'
if len(sys.argv) >= 2:
  size = int(sys.argv[1])
if len(sys.argv) >= 3:
  mode = sys.argv[2]

def getState():
  if mode == 'random':
    return random.randint(0, 1)
  else:
    return 0

frame = [[getState() for j in range(size)] for i in range(size)]

filename = f"{mode}_{size}.txt"
with open(filename, "w") as f:
  for row in frame:
    f.write(" ".join(str(x) for x in row) + "\n")
