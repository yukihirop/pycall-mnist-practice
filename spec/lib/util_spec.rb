require 'spec_helper'

describe Util do
  describe '.identity' do
    let(:x) { 1 }
    subject { Util.identity(x) }
    it { should eq x }
  end

  describe '.sigmoid' do
    let(:x) { NP.array([-1.0, 1.0, 2.0]) }
    let(:answer) { [0.26894142, 0.73105858, 0.88079708] }
    subject { np_array_to_a(Util.sigmoid(x), 8) }
    it { should eq answer }

    context 'when batch' do
      let(:x2) { NP.array([x, x]) }
      subject { Util.sigmoid(x2) }
      it { expect(np_array_to_a(subject[0], 8)).to eq answer }
      it { expect(np_array_to_a(subject[1], 8)).to eq answer }
    end
  end

  describe '.sigmoid_grad' do
    let(:x) { NP.array([-1.0, 1.0, 2.0]) }
    subject { np_array_to_a(Util.sigmoid_grad(Util.sigmoid(x)), 8) }
    it { should eq [0.19661193, 0.19661193, 0.10499359] }
  end

  describe '.softmax' do
    let(:x) { NP.array([0.3, 2.9, 4.0]) }
    let(:answer) { [0.01821127, 0.24519181, 0.73659691] }
    subject { np_array_to_a(Util.softmax(x), 8) }
    it { should eq answer }

    context 'when batch' do
      let(:x2) { NP.array([x, x]) }
      subject { Util.softmax(x2) }
      it { expect(np_array_to_a(subject[0], 8)).to eq answer }
      it { expect(np_array_to_a(subject[1], 8)).to eq answer }
    end
  end

  describe '.cross_entropy_error' do
    let(:t) { NP.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) }
    let(:y) {NP.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]) }
    subject { Util.cross_entropy_error(y, t).round(12) }
    let(:answer) { 0.510825623766 }
    it { should eq answer }

    context 'when batch' do
      let(:t2) { NP.array([t, t]) }
      let(:y2) { NP.array([y, y]) }
      subject { Util.cross_entropy_error(y2, t2).round(12) }
      it { should eq answer }
    end
  end

  describe '.numerical_gradient'  do
    let(:f) { ->(x) { x[0]**2 + x[1] ** 2 } }
    let(:x) { NP.array([3.0, 4.0]) }
    subject { np_array_to_a(Util.numerical_gradient(f, x), 1) }
    it { should eq [6.0, 8.0]}
  end
end
