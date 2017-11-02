require 'bundler'
Bundler.require

require 'pycall/import'
include PyCall::Import

pyimport 'matplotlib.pyplot', as: :plt

xs = [*1..100].map {|x| (x - 50) * Math::PI / 100.0 }
ys = xs.map {|x| Math.sin(x) }


binding.pry

plt.plot(xs, ys)
plt.show()
