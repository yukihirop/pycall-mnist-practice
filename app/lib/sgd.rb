# http://li.nu/blog/2016/03/handwritten-number-classification-with-caffe-part3.html

# 確率的勾配降下法(SGD)
class SGD
  def update(network, grads, learning_rate)
    network.params = network.params.map do |key, value|
      # w(重み) <- w(重) - n*(dE/dw)
      # こんな感じのやつかな？
      # keyは各ノードの識別子
      # valueは各ノードの重み
      [key, value - learning_rate * grads[key]]
    end.to_h
  end
end
