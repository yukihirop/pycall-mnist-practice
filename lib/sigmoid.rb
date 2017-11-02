require './lib/util'

class Sigmoid
  def forward(x)
    self.out = Util.sigmoid(x)
  end

  def backward(dout)
    dout * Util.sigmoid_grad(out)
  end

  private

  attr_accessor :out
end
