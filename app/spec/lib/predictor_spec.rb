require 'spec_helper'
require 'network'

describe Predictor do
  let(:image) { ChunkyPNG::Image.from_file('spec/fixtures/canvas.png') }
  let(:data_url) { image.to_data_url }
  let(:pixels) { described_class.parse(data_url)[1] }


  describe '.parse' do
    let(:network) { Network.new }
    let(:network_input_size)  { network.input_size }
    subject { pixels.size }
    it { should eq network_input_size }
  end

  describe '.predict' do
    let(:predictor) { described_class.new('data/params.json') }
    subject { predictor.predict(pixels).argmax().to_s.to_i }
    it { should eq 2 }
  end
end
