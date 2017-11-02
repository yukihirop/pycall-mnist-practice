require './lib/np'

# ほんとコレ！
# http://pythonskywalker.hatenablog.com/entry/2016/12/25/144926

# 誤差逆伝搬法
# http://www.yukisako.xyz/entry/backpropagation
# softmaxレイヤーは確率にするためにあるレイヤー
# http://nihemak.hatenablog.com/entry/2016/07/02/205330
class Affine
  attr_accessor :dw, :db

  def initialize(w, b)
    self.w = w # 重み
    self.b = b # バイアス
  end

  def forward(x)
    self.x = x # 入力
    # 入力*重み + バイアス
    # http://pythonskywalker.hatenablog.com/entry/2016/12/25/144926
    NP.dot(x, w) + b
  end

  # 逆伝搬
  def backward(dout)
    # 重みの形状を転地を行って、それをdoutでdotする
    dx = NP.dot(dout, w.T)
    # 入力の形状の転置を行って、それをdoutでdotする
    self.dw = NP.dot(x.T, dout)
    # バイアスはaxis=0で微分する
    # [[1,2,3],[100,200,300]]→[101,202,303]
    self.db = NP.sum(dout, 0)
    dx
  end

  private

  attr_accessor :w, :b, :x
end
