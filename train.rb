require 'bundler'
Bundler.require

require '../app/lib/network'
require '../app/lib/trainer'
require '../app/lib/predictor'
require '../app/lib/np'

require 'benchmark'
require 'pycall/import'
include PyCall::Import

pyimport 'matplotlib.pyplot', as: :plt

limit = ARGV[0].nil? ? 100 : ARGV[0].to_i
iterms_num = ARGV[1].to_i.nonzero? || 500
batch_size = ARGV[2].to_i.nonzero? || 10
path =ARGV[3] || 'tmp/params.json'

train_acc_list = []
test_acc_list = []

memprof = true
plot = true

network = if File.exist?(path)
            Predictor.new(path).network
          else
            Network.new
          end

Memprof2.start if memprof

result = Benchmark.measure do
  trainer = Trainer.new(network)
  # 何も指定しなかったらiterms_num: 500
  # 何も指定しなかったらbatch_size: 10
  _, train_acc_list, test_acc_list = trainer.train_mnist(limit, iterms_num, batch_size, 0.1, plot)
  # tmp/params.jsonにW1・b1・W2・b2のデータが保存される
  Trainer.save_params(path, network.params)
end

puts Benchmark::CAPTION
puts result

puts "(train acc, test acc) = (#{train_acc_list.last * 100}%, #{test_acc_list.last * 100}%)"

# ベンチマークを保存
if memprof
  Memprof2.report(out: 'tmp/memprof2_report')
  Memprof2.stop
end

if plot
  puts "結果を表示します"
  size = train_acc_list.size.to_i
  puts "train_acc_listのサイズは、#{size}です"

  # x = NP.arange(2)

  x = [1,2]

  # うまく表示できない。
  # 引数にはrubyのArray型の配列をわたす
  plt.plot(x, train_acc_list)
  plt.plot(x, test_acc_list)
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.ylim(0, 1.0)
  plt.legend('lower right')
  plt.show()
end
