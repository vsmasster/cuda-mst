import os
import subprocess
from graph import Graph
from plot import Plot

test_cases = [
   #('losowy_rzadki', Graph.generate_random_graph, [(40, 100, 100), (100, 10000, 100), (1000, 100000, 100), (10000, 50000, 100), (50000, 1000000, 100), (100000, 5000000, 100), (300000, 6000000, 100), (500000, 7000000, 100), (700000, 8000000, 100), (850000, 9000000, 100), (1000000, 10000000, 100)]),
   #('losowy_gesty', Graph.generate_random_graph, [(20, 100, 100), (100, 5000, 100), (500, 10000, 100), (1000, 100000, 100), (3000, 300000, 100), (5000, 1000000, 1000), (10000, 5000000, 1000), (20000, 7000000, 1000), (30000, 10000000, 1000)]),
   ('klika', Graph.generate_clique, [(10, 1000), (100, 1000), (500, 1000), (1000, 1000), (2000, 1000), (3000, 1000), (4000, 1000), (5000, 1000)]),
]

os.system("cd ../source; make")

for test in test_cases:
   test_name, generate_func, test_conditions = test

   print "Running: " + test_name
   plot = Plot(test_name, ['cpu', 'dpk', 'vi'], 'm', 'ms')
   it = 0

   for condition in test_conditions:
      n = condition[0]
      m = 0

      if test_name == "klika":
         m = n*(n-1) / 2
      else:
         m = condition[1]

      file = open("in" + str(it), "w+")
      file.write(Graph.to_string(generate_func(condition)))
      file.close()

      print "Test for n = " + str(condition[0]) + " generated"
      os.system("cd ../source; ./main < in" + str(it) + " > out")

      file = open("out", "r+")
      cpu = file.readline().split(" ")[1].rstrip()
      vi = file.readline().split(" ")[1].rstrip()
      dpk = file.readline().split(" ")[1].rstrip()

      plot.add_result("cpu", m, cpu)
      plot.add_result("vi", m, vi)
      plot.add_result("dpk", m, dpk)
      print cpu + " " + vi + " " + dpk

      file.close()

      it += 1

   plot.show()
   plot.close()


os.system("make clean")
os.system("rm in*")
os.system("rm out")
