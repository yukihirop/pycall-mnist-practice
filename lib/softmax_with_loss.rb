require './lib/util'

# Softmaxはある事象が起こる確率を
# WithLossで誤差関数を
# SoftWithMaxは誤差逆伝搬する際に必要な誤差関数を表す。
# 入力→確率分布→誤差関数
class SoftmaxWithLoss
  def forward(x, t)
    self.t = t
    self.y = Util.softmax(x)


    self.loss = Util.cross_entropy_error(y,t)
  end

  def backward(_)
    batch_size = t.shape[0]
    (y-t) / batch_size
  end

  private

  attr_accessor :loss, :y, :t
end
