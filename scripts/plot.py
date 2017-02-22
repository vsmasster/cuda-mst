import matplotlib.pyplot as plt

class Plot:
   def __init__(self, plot_name, keys, xlabel, ylabel):
      plt.ylabel = ylabel
      plt.xlabel = xlabel

      self.results = {}

      for key in keys:
         self.results[key] = []

      self.plot_name = plot_name

   def add_result(self, key, x, y):
      self.results[key].append((x, y))

   def show(self):
      for key in self.results.keys():
         xs, ys = [], []

         for x, y in self.results[key]:
            xs.append(x)
            ys.append(y)

         plt.plot(xs, ys, label=key)

      plt.legend(loc=2)
      plt.savefig(self.plot_name + ".jpg")

   def close(self):
      plt.close()
