require './lib'

require 'benchmark'

predicator = Predicator.new('data/params.json')

limit = 10
# テストで訓練ではない。なので訓練の引数は_
# limit次元に加工される
_, _, x_test, t_test = Loader.load_mnist(false, limit, true, false)

result = Benchmark.measure do
  limit.times do |i|
    # 各layersを繰り返し伝搬していくpredictor.predict
    # 一番大きい要素を返しているx_testは10次元ベクタデータ
    y = predictor.predict(x_test[i]).argmax.().to_s
    t = t_test[i].to_s
    puts "#{y} == #{t}: #{y == t ? 'o' : 'x' }"
  end
end

puts Benchmark::CAPTION
puts result
