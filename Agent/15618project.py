import os
import numpy as np
import random

grid_size = 100
init_size = 10
sim_cycles = 100
num_model = 10


class Network:
  def __init__(self, weights_1=None, weights_2=None, weights_3=None, weights_4=None, biases=None):
    self.network_outputs = int(init_size * init_size)
    self.layer_1 = init_size * init_size
    self.layer_2 = init_size * init_size
    self.layer_3 = init_size * init_size
    if weights_1 is None:
      weights_1_shape = (self.layer_1, init_size * init_size)
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
    if biases is None:
      self.biases = np.random.normal(size=(self.network_outputs))
    else:
      self.biases = biases

  def clone(self):
    return Network(np.copy(self.weights_1), np.copy(self.weights_2), np.copy(self.weights_3), np.copy(self.weights_4), np.copy(self.biases), self.type)

  def forward(self, observations=np.ones(init_size * init_size)):
    outputs_sub1 = np.matmul(self.weights_1, observations)
    outputs_sub2 = np.matmul(self.weights_2, outputs_sub1)
    outputs_sub3 = np.matmul(self.weights_3, outputs_sub2)
    outputs = np.add(
        np.matmul(self.weights_4, outputs_sub3), self.biases)
    for i in range(len(outputs)):
      if outputs[i] > 0:
        outputs[i] = 1
      else:
        outputs[i] = 0

    return outputs

  def copy_and_mutate(self, network, mr=0.1):
    self.weights_1 = np.add(
        network.weights_1, np.random.normal(size=self.weights_1.shape) * mr)
    self.weights_2 = np.add(
        network.weights_2, np.random.normal(size=self.weights_2.shape) * mr)
    self.weights_3 = np.add(
        network.weights_3, np.random.normal(size=self.weights_3.shape) * mr)
    self.weights_4 = np.add(
        network.weights_4, np.random.normal(size=self.weights_4.shape) * mr)
    self.biases = np.add(
        network.biases, np.random.normal(size=self.biases.shape) * mr)

  def copy(self, network):
    self.weights_1 = network.weights_1
    self.weights_2 = network.weights_2
    self.weights_3 = network.weights_3
    self.weights_4 = network.weights_4
    self.biases = network.biases


class Enviroment:
  def __init__(self, size=grid_size, init=init_size, starting=None, cycles=sim_cycles):
    self.grid_size = size
    self.init_size = init
    self.grid = [[0 for col in range(self.grid_size)]
                 for row in range(self.grid_size)]
    self.middle = int(self.grid_size/2 - init_size/2)
    self.sim_cycles = cycles
    self.cur_cycles = 0
    self.max_cnt = -1
    if starting is None:
      for row in range(self.init_size):
        for col in range(self.init_size):
          self.grid[self.middle+row][self.middle+col] = random.randint(0, 1)
    else:
      for row in range(self.init_size):
        for col in range(self.init_size):
          self.grid[self.middle+row][self.middle +
                                     col] = starting[row*self.init_size + col]

  def step(self):
    total_cnt = 0
    new_grid = [[0 for col in range(self.grid_size)]
                for row in range(self.grid_size)]
    for row in range(self.grid_size):
      for col in range(self.grid_size):
        neighbor_cnt = 0
        for rr in range(row-1, row+2):
          for cc in range(col-1, col+2):
            if rr == row and cc == col:
              continue
            if self.grid[rr % self.grid_size][cc % self.grid_size] == 1:
              neighbor_cnt = neighbor_cnt + 1

        if (neighbor_cnt == 3) or (neighbor_cnt == 3 and self.grid[row][col] == 1):
          new_grid[row][col] = 1
          total_cnt = total_cnt + 1
        else:
          new_grid[row][col] = 0

    self.grid = new_grid
    return total_cnt

  def simulate(self):
    max_step = -1
    cur_cnt = 0
    for step in range(self.sim_cycles):
      cur_cnt = self.step()
      if cur_cnt > self.max_cnt:
        self.max_cnt = cur_cnt
        max_step = step + 1

    return self.max_cnt/max_step + cur_cnt


def printGrid(grid):
  for row in grid:
    print(row)


models = []
for i in range(num_model):
  models.append(Network())

episodes = 5
for episode in range(1, episodes+1):
  highest_score = 0.0
  best_model = -1
  for i in range(len(models)):
    env = Enviroment(starting=models[i].forward())
    score = env.simulate()
    if score > highest_score:
      highest_score = score
      best_model = i

  print('Episode:{} Best Model:{} Best Score:{}'.format(
      episode, best_model, highest_score))

  for i in range(len(models)):
    if i == 0:
      models[i].copy(network=models[best_model])
    elif i < (7):
      models[i].copy_and_mutate(network=models[best_model])
    elif i < (9):
      models[i].copy_and_mutate(network=models[best_model], mr=0.2)
    else:
      models[i].copy_and_mutate(network=models[best_model], mr=0.5)
