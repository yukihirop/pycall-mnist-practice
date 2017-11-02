require './lib/trainer'
require './lib/network'
require './lib/np'

class Predictor
  attr_accessor :network

  class << self
    def parse(data)
      length = 28
      # canvasをインポート
      canvas = ChunkyPNG::Canvas.from_data_url(data).resize(length, length)
      pixels = NP.array(Array.new(length) do |i|
        Array.new(length) do |j|
          canvas[j, i] / 255.0
        end
      end.flatten)
      # pixcelsは正規化した値？？
      [canvas, pixels]
    end
  end


  def initialize(path)
    params = Trainer.load_params(path)
    self.network = Network.new
    network.params = params
  end

  def predict(x)
    network.predict(x)
  end
end
