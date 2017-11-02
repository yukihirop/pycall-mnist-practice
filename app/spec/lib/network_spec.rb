require 'spec_helper'
require 'network'

describe Network do
  let(:network) do
    Network.new.tap do |n|
      params = n.params
      params[:W1].fill(0.01)
      n.hidden_size.times do |i|
        n.output_size.times do |j|
          params[:W2][i][j] = j * 0.01
        end
      end
      n.params = params
    end
  end

  let(:limit) { 2 }
  let(:data) { Loader.load_mnist(true, limit) }
  let(:x_train) { data[0] }
  let(:t_train) { data[1] }
  let(:x_test) { data[2] }
  let(:t_test) { data[3] }
  let(:digits) { 6 }

  describe '#predict' do
    let(:answer) { [0.0, 0.53970589, 1.07941178, 1.61911766, 2.15882355, 2.69852944, 3.23823533, 3.77794121, 4.3176471, 4.85735299].map { |a| a.round(digits) } }
    subject { network.predict(x_train[0]) }

    it { expect(np_array_to_a(subject, digits)).to eq answer }

    context 'when batch' do
      let(:answer2) { [0.0, 0.60970589, 1.21941178, 1.82911766, 2.43882355, 3.04852944, 3.65823533, 4.26794122, 4.8776471, 5.48735299].map { |a| a.round(digits) } }
      subject { network.predict(x_train[NP.arange(2)]) }

      it { expect(np_array_to_a(subject[0], digits)).to eq answer }
      it { expect(np_array_to_a(subject[1], digits)).to eq answer2 }
    end
  end

  describe '#loss' do
    subject { network.loss(x_train, t_train).round(7) }
    let(:loss) { 4.64902540714.round(7) }

    it { should eq loss }
  end

  describe '#accuracy' do
    let(:x) { x_train[NP.arange(2)] }
    let(:t) { NP.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]) }
    let(:y) { NP.array([[0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) }
    before do
      allow(network).to receive(:predict).and_return(y)
    end
    subject { network.accuracy(x,t) }
    it { should eq 0.5 }
  end

  describe '#gradient' do
    let(:train_size) { x_train.shape[0] }
    let(:batch_size) { 1 }
    let(:batch_mask) { NP.choice(train_size, batch_size) }
    let(:x_batch)    { x_train[batch_mask] }
    let(:t_batch)    { t_train[batch_mask] }

    let(:numerical_gradient) { network.numerical_gradient(x_batch, t_batch) }
    let(:numerical_b1)       { np_array_to_a(numerical_gradient[:b1], 8) }
    subject { np_array_to_a(gradient[:b1], 8) }

    it { should eq numerical_b1 }
  end
end
