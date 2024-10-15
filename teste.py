from cec2013lsgo.cec2013 import Benchmark
from numpy.random import rand
from utils import generatePopulation

pop = generatePopulation(905, 500, -100, 100)
print(pop[0])

bench = Benchmark()
# print(bench.get_info(1))

info = bench.get_info(1)
dim = info['dimension']
print(f"dim: {dim}")
sol = info['lower']+rand(10)*(info['upper']-info['lower'])

fun_fitness = bench.get_function(15)
print(fun_fitness(pop[0]))
# print(sol)