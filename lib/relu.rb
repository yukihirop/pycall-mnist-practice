# http://nihemak.hatenablog.com/entry/2016/07/02/205330
# reluはy=x(x>=0); y=0 (x<0);な関数
class Relu

  # doutはNArray??
  def forward(x)
    self.mask = x <= 0

    out = x.copy()
    out[mask] = 0
    out
  end

  def backward(dout)
    dout[mask] = 0
    dout
  end

  attr_accessor :mask
end
