import random

class Graph:
   @staticmethod
   def generate_random_graph(input):

      N, M, W = input

      graph = []
      graph.append((N, M))

      m = M

      for u in range(1, N):
         v = random.randint(0, u - 1)
         w = random.randint(1, W)
         graph.append((u, v, w))
         m -= 1

      while m > 0:
         u = random.randint(0, N-1)
         v = random.randint(0, N-1)
         w = random.randint(1, W)

         if u != v:
            graph.append((u, v, w))
            m -= 1
            
      return graph

   @staticmethod
   def generate_clique(input):
      N, W = input

      graph = []
      graph.append((N, N*(N-1) / 2))

      for u in range(0, N):
         for v in range(u+1, N):
            w = random.randint(1, W)
            graph.append((u, v, w))

      return graph

   @staticmethod
   def generate_lolipop(input):
      N, W, C = input
      
      graph = []
      graph.append((N, C*(C-1)/2 + N - C))

      for u in range(0, C):
         for v in range(u+1, C):
            w = random.randint(1, W)
            graph.append((u, v, w))

      u = C

      for v in range(C, N):
         w = random.randint(1, W)
         graph.append((u, v, w))
         u = v

      return graph

   @staticmethod
   def GraphGenerator(conditions):
      for func, args in conditions:
         yield func(args)

   @staticmethod
   def to_string(graph):
      res = str(graph[0][0]) + " " + str(graph[0][1]) + "\n"
      graph = graph[1:]

      for u,v,w in graph:
         res += str(u) + " " + str(v) + " " + str(w) + "\n"

      return res.rstrip()
