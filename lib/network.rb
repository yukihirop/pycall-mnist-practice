require './lib/np'
require './lib/affine'
require './lib/relu'
require './lib/softmax_with_loss'

class Network
  attr_accessor :params, :input_size, :hidden_size, :output_size, :layers, :lastLayer, :current_layer

  def initialize(input_size = 784, hidden_size = 50, output_size = 10, weight_init_std = 0.01)
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.current_layer = nil

    init_params(weight_init_std)
  end

  # ネットワークのレイヤーを通す
  # 出力ｙを予測する
  def predict(x)
    y = x
    layers.values.each do |layer|
      self.current_layer = layer
      y = layer.forward(y)
    end
    y
  end

  # 誤差率
  def loss(x, t)
    y = predict(x)
    lastLayer.forward(y, t)
  end

  # 正答率
  # In [8]: a
  # Out[8]:
  #   array([[5, 3],
  #          [8, 9]])
  #
  # In [9]: a.shape
  # Out[9]: (2, 2)
  #
  # In [10]: a.shape = (4, 1)
  #
  # In [11]: a
  # Out[11]:
  #   array([[5],
  #          [3],
  #          [8],
  #          [9]])
  #
  def accuracy(x, t)
    y = predict(x)
    # yの出力列で値が最大のindex
    y = NP.argmax(y, 1)
    t = NP.argmax(t, 1)

    # x.shape[0]は何行か？
    # sum(y == t) は tにyが含まれていたらたす
    NP.sum(y == t) / x.shape[0].to_f
  end

  # 逆誤差伝搬法で求める
  def numerical_gradient(x,t)
    {
      W1: Util.numerical_gradient(loss_w(:W1, x, t), w1),
      b1: Util.numerical_gradient(loss_w(:b1, x, t), b1),
      W2: Util.numerical_gradient(loss_w(:W2, x, t), w2),
      b2: Util.numerical_gradient(loss_w(:b2, x, t), b2),
    }
  end

  def loss_w(key, x, t)
    # 何故プロックにする必要があるん？
    lambda do |w|
      tmp_params = params
      self.params = params.merge(key => w)
      # self.paramsを書き換えた上で実行
      l = loss(x,t)
      self.params = tmp_params
    end
  end

  def gradient(x, t)
    gradient_delta(x, t)
    {
      W1: layers[:affine1].dw,
      b1: layers[:affine1].db,
      W2: layers[:affine2].dw,
      b2: layers[:affine2].db
    }
  end

  # paramsをセットするだけではない
  def params=(params)
    @params = params
    init_layers
  end

  private

  def gradient_delta(x,t)
    loss(x,t)

    dout = 1
    ([lastLayer] + layers.values.reverse).each do |layer|
      dout = layer.backward(dout)
    end
  end

  def w1
    params[:W1]
  end

  def w2
    params[:W2]
  end

  def b1
    params[:b1]
  end

  def b2
    params[:b2]
  end

  # 重みの標準偏差
  def init_params(weight_init_std)
    self.params = {
      # input_size(784)行×hidden_size(50)列の配列
      W1: weight_init_std * NP.randn(input_size, hidden_size),
      # 全部0.の50列の配列
      b1: NP.zeros(hidden_size),
      W2: weight_init_std * NP.randn(hidden_size, output_size),
      b2: NP.zeros(output_size)
    }
    self.params
  end

  def init_layers
    self.layers = {
      affine1: Affine.new(w1, b1),
      rele1: Relu.new,
      affine2: Affine.new(w2, b2)
    }
    self.lastLayer = SoftmaxWithLoss.new
  end
end
