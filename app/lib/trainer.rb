require './lib/sgd'
require './lib/loader'
require 'json'

# CNNを訓練するデータをロードして来る所
class Trainer
  class << self
    def save_params(path, params)
      json = params.map do |key, value|
        params_to_json(key, value)
      end.to_h
      # json形式で保存するのかな
      File.write(path, json.to_json)
    end

    def load_params(path)

      JSON.parse(File.read(path)).map { |k,v| [k.to_sym, NP.array(v)] }.to_h
    end

    private

    def params_to_json(key, value)
      # narrayの次元が一次元の場合
      if value.ndim == 1
        [ key, Util.np_array_to_a(value) ]
      else
        [ key, Array.new(value.shape[0]) { |i| Util.np_array_to_a(value[i]) } ]
      end
    end
  end

  # optimizerは最適化する
  # SGDで最適化する
  # 確率的勾配硬化法を使う
  def initialize(network)
    self.network = network
    self.optimizer = SGD.new
  end

  # mnistで訓練する
  #
  def train_mnist(limit, iterms_num, batch_size, learning_rate = 0.1, listing = true)
    # limit次元に絞られたmnist
    mnist = Loader.load_mnist(true, limit, listing)
    x_train = mnist[0]

    train_size = x_train.shape[0]

    # 訓練によるロス
    # 訓練による正答率
    # テストによる正答率
    list = { train_loss: [], train_acc: [], test_acc: []}

    # batch_size 訓練するデータから適当にとった数
    # train_size 1エポック(訓練する全データの数)
    iter_per_epoch = [train_size / batch_size, 1].max

    train_batch(iterms_num, learning_rate, batch_size, train_size, mnist) do |i, x_batch, t_batch|
      # 誤差listに出力
      loss_to_list(list, x_batch, t_batch) if listing
      # iter_per_epochの倍数のときに正答率をlistに出すようにする
      acc_to_list(list, mnist) if listing && (i % iter_per_epoch).zero?
    end

    list.values
  end

  private

  attr_accessor :network, :optimizer

  def batch(mnist, batch_size, train_size)
    x_train, t_train = mnist
    # 訓練サイズtrain_size以下の数字でbach_size次元の数
    batch_mask = NP.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    [x_batch, t_batch]
  end

  # xは認識対象の入力データ
  # tは教師データ
  def train_batch(iterms_num, learning_rate, batch_size, train_size, mnist)
    progressbar = ::ProgressBar.create(total: iterms_num, format: '%c / %C')

    iterms_num.times do |i|
      # mnistから入力xと訓練tを取り出して、
      x_batch, t_batch = batch(mnist, batch_size, train_size)

      # 伝搬すべき誤差
      grads = network.gradient(x_batch, t_batch)

      optimizer.update(network, grads, learning_rate)

      # これはなんだ？？
      yield i, x_batch, t_batch

      # バーをすすめる
      progressbar.increment
    end
  end

  # x_batchってなんだろう
  def loss_to_list(list, x_batch, t_batch)
    loss = network.loss(x_batch, t_batch)
    list[:train_loss] << loss
  end

  def acc_to_list(list, mnist)
    x_train, t_train, x_test, t_test = mnist

    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    list[:train_acc] << train_acc
    list[:test_acc] << test_acc
  end
end
