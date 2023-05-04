import os
import time
import numpy as np
import random
from cpu_renderer_wrapper import PyCpuRenderer

grid_size = 20
init_size = 10
sim_cycles = 1000
num_place = 25
num_model = 25
episodes = 1000


class Network:
  def __init__(self, init_size=init_size, weights_1=None, weights_2=None, weights_3=None, weights_4=None, biases_in=None, biases_out=None):
    self.init_size = init_size
    self.network_outputs = 2 * self.init_size
    self.layer_1 = 2 * self.init_size * self.init_size
    self.layer_2 = self.init_size * self.init_size
    self.layer_3 = self.init_size * self.init_size

    if weights_1 is None:
      weights_1_shape = (self.layer_1, 2 * self.init_size * self.init_size)
      self.weights_1 = np.random.normal(size=weights_1_shape)
    else:
      self.weights_1 = weights_1
    if weights_2 is None:
      weights_2_shape = (self.layer_2, self.layer_1)
      self.weights_2 = np.random.normal(size=weights_2_shape)
    else:
      self.weights_2 = weights_2
    if weights_3 is None:
      weights_3_shape = (self.layer_3, self.layer_2)
      self.weights_3 = np.random.normal(size=weights_3_shape)
    else:
      self.weights_3 = weights_3
    if weights_4 is None:
      weights_4_shape = (self.network_outputs, self.layer_3)
      self.weights_4 = np.random.normal(size=weights_4_shape)
    else:
      self.weights_4 = weights_4
    if biases_in is None:
      self.biases_in = np.random.normal(size=(self.layer_1))
    else:
      self.biases_in = biases_in
    if biases_out is None:
      self.biases_out = np.random.normal(size=(self.network_outputs))
    else:
      self.biases_out = biases_out

  def clone(self):
    return Network(np.copy(self.weights_1), np.copy(self.weights_2), np.copy(self.weights_3), np.copy(self.weights_4), np.copy(self.biases_in), np.copy(self.biases_out), self.type)

  def forward(self, observations=np.zeros(2 * init_size * init_size)):
    outputs_sub0 = np.add(observations, self.biases_in)
    outputs_sub1 = np.matmul(self.weights_1, outputs_sub0)
    outputs_sub2 = np.matmul(self.weights_2, outputs_sub1)
    outputs_sub3 = np.matmul(self.weights_3, outputs_sub2)
    outputs = np.add(np.matmul(self.weights_4, outputs_sub3), self.biases_out)

    while True:
      rows = outputs[:self.init_size]
      row = np.argmax(rows)
      cols = outputs[self.init_size:]
      col = np.argmax(cols)
      if observations[row * self.init_size + col] == 0:
        break

      if rows[row] > 0 and cols[col] > 0:
        rows[row] = 0
        cols[col] = 0
      else:
        while True:
          row = random.randint(0, self.init_size - 1)
          col = random.randint(0, self.init_size - 1)
          if observations[row * self.init_size + col] == 0:
            break

        break

    return row, col

  def copy_and_mutate(self, network, mr=0.1):
    self.weights_1 = np.add(
        network.weights_1, np.random.normal(size=self.weights_1.shape) * mr)
    self.weights_2 = np.add(
        network.weights_2, np.random.normal(size=self.weights_2.shape) * mr)
    self.weights_3 = np.add(
        network.weights_3, np.random.normal(size=self.weights_3.shape) * mr)
    self.weights_4 = np.add(
        network.weights_4, np.random.normal(size=self.weights_4.shape) * mr)
    self.biases_in = np.add(
        network.biases_in, np.random.normal(size=self.biases_in.shape) * mr)
    self.biases_out = np.add(
        network.biases_out, np.random.normal(size=self.biases_out.shape) * mr)

  def copy(self, network):
    self.weights_1 = network.weights_1
    self.weights_2 = network.weights_2
    self.weights_3 = network.weights_3
    self.weights_4 = network.weights_4
    self.biases_in = network.biases_in
    self.biases_out = network.biases_out


class Enviroment:
  def __init__(self, size=grid_size, init=init_size, starting_left=None, starting_right=None, cycles=sim_cycles):
    self.grid_size = size
    self.init_size = init
    self.grid = [[0 for col in range(self.grid_size)]
                 for row in range(self.grid_size)]
    self.obs = [0 for i in range(2 * self.init_size * self.init_size)]
    self.middle = int(self.grid_size/2)
    self.sim_cycles = cycles

  def place(self, player, row, col):
    if player == 0:
      start = self.middle - self.init_size
      self.grid[start+row][start+col] = 1
      self.obs[row * self.init_size + col] = 1
    else:
      start = self.middle
      self.grid[start+row][start+col] = 2
      self.obs[self.init_size * self.init_size +
               row * self.init_size + col] = 1

    return self.obs

  def step(self):
    left_total_cnt = 0
    right_total_cnt = 0
    new_grid = [[0 for col in range(self.grid_size)]
                for row in range(self.grid_size)]
    for row in range(self.grid_size):
      for col in range(self.grid_size):
        left_neighbor_cnt = 0
        right_neighbor_cnt = 0
        for rr in range(row-1, row+2):
          for cc in range(col-1, col+2):
            if rr == row and cc == col:
              continue
            if self.grid[rr % self.grid_size][cc % self.grid_size] == 1:
              left_neighbor_cnt = left_neighbor_cnt + 1
            elif self.grid[rr % self.grid_size][cc % self.grid_size] == 2:
              right_neighbor_cnt = right_neighbor_cnt + 1

        if self.grid[row][col] == 0:
          if (left_neighbor_cnt == 3) and (right_neighbor_cnt != 3):
            new_grid[row][col] = 1
            left_total_cnt = left_total_cnt + 1
          elif (right_neighbor_cnt == 3) and (left_neighbor_cnt != 3):
            new_grid[row][col] = 2
            right_total_cnt = right_total_cnt + 1
          else:
            new_grid[row][col] = 0
        elif self.grid[row][col] == 1:
          if (left_neighbor_cnt == 2):
            new_grid[row][col] = 1
            left_total_cnt = left_total_cnt + 1
          elif (right_neighbor_cnt == 3):
            new_grid[row][col] = 2
            right_total_cnt = right_total_cnt + 1
          else:
            new_grid[row][col] = 0
        else:
          if (right_neighbor_cnt == 2):
            new_grid[row][col] = 2
            right_total_cnt = right_total_cnt + 1
          elif (left_neighbor_cnt == 3):
            new_grid[row][col] = 1
            left_total_cnt = left_total_cnt + 1
          else:
            new_grid[row][col] = 0

    self.grid = new_grid
    return left_total_cnt, right_total_cnt

  def simulate(self):
    left_max_cnt = -1
    right_max_cnt = -1
    left_max_step = -1
    right_max_step = -1
    left_cur_cnt = 0
    right_cur_cnt = 0
    for step in range(self.sim_cycles):
      left_cur_cnt, right_cur_cnt = self.step()
      if left_cur_cnt == 0 and right_cur_cnt == 0:
        break
      if left_cur_cnt > left_max_cnt:
        left_max_cnt = left_cur_cnt
        left_max_step = step + 1
      if right_cur_cnt > right_max_cnt:
        right_max_cnt = right_cur_cnt
        right_max_step = step + 1

    return left_max_cnt/left_max_step + left_cur_cnt, right_max_cnt/right_max_step + right_cur_cnt


def printGrid(grid):
  for row in grid:
    print(row)


weights_1 = np.load('./Result/weights_1.npy') if os.path.isfile(
    './Result/weights_1.npy') else None
weights_2 = np.load('./Result/weights_2.npy') if os.path.isfile(
    './Result/weights_2.npy') else None
weights_3 = np.load('./Result/weights_3.npy') if os.path.isfile(
    './Result/weights_3.npy') else None
weights_4 = np.load('./Result/weights_4.npy') if os.path.isfile(
    './Result/weights_4.npy') else None
biases_in = np.load('./Result/biases_in.npy') if os.path.isfile(
    './Result/biases_in.npy') else None
biases_out = np.load('./Result/biases_out.npy') if os.path.isfile(
    './Result/biases_out.npy') else None

models = []
for i in range(num_model):
  models.append(Network(weights_1=weights_1, weights_2=weights_2, weights_3=weights_3,
                weights_4=weights_4, biases_in=biases_in, biases_out=biases_out))

startTime = time.time()
for episode in range(1, episodes+1):
  highest_score = 0.0
  best_model = -1
  for i in range(len(models)):
    env = Enviroment()
    obs = env.obs
    for j in range(2 * num_place):
      if (j % 2 == 0):
        row, col = models[0].forward(obs)
        obs = env.place(0, row, col)
        obs = obs[init_size*init_size:] + obs[:init_size*init_size]
      else:
        row, col = models[i].forward(obs)
        obs = env.place(1, row, col)

    _, score = env.simulate()
    if score > highest_score:
      highest_score = score
      best_model = i

  if best_model == -1:
    best_model = 0

  print('Episode:{} Best Model:{} Best Score:{}'.format(
      episode, best_model, highest_score))

  for i in range(len(models)):
    if i == 0:
      models[i].copy(network=models[best_model])
    elif i < (0.7*len(models)):
      models[i].copy_and_mutate(network=models[best_model])
    elif i < (0.9*len(models)):
      models[i].copy_and_mutate(network=models[best_model], mr=0.3)
    else:
      models[i].copy_and_mutate(network=models[best_model], mr=0.7)

endTime = time.time()

print("Training: " + str(endTime - startTime) + "seconds")
env = Enviroment()
obs = env.obs
for i in range(2 * num_place):
  if (i % 2 == 0):
    row, col = models[0].forward(obs)
    obs = env.place(0, row, col)
    obs = obs[init_size*init_size:] + obs[:init_size*init_size]
  else:
    row, col = models[0].forward(obs)
    obs = env.place(1, row, col)

print("Placing")
printGrid(env.grid)
left_score, right_score = env.simulate()
print("Left Score: " + str(left_score) + "; Right Score: " + str(right_score))
printGrid(env.grid)

folder = "Result"
os.mkdir(folder)
np.save(folder + '/weights_1.npy', models[0].weights_1)
np.save(folder + '/weights_2.npy', models[0].weights_2)
np.save(folder + '/weights_3.npy', models[0].weights_3)
np.save(folder + '/weights_4.npy', models[0].weights_4)
np.save(folder + '/biases_in.npy', models[0].biases_in)
np.save(folder + '/biases_out.npy', models[0].biases_out)
