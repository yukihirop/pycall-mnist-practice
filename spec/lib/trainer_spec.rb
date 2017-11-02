require 'spec_helper'
require 'network'

describe Trainer do
  let(:network) { Network.new }

  # 帰納法みたいなテストの書き方
  describe  '#train_mnist' do
    # limit: 100件のデータに対して
    # batch_size: 10件づつデータを渡して
    # iter_per_epoch: 10回イテレートを繰り返して
    # 1エポック
    let(:limit)         { 100 }
    let(:iter_num)      { 10 }
    let(:batch_size)    { 10 }
    let(:learning_rate) { 0.1}
    let(:listing)       { true  }
    let(:trainer)       { Trainer.new(network) }
    let(:data)          { trainer.train_mnist(limit, iter_num, batch_size) }
    let(:train_loss_list) { data[0] }
    let(:train_acc_list)  { data[1] }
    let(:test_acc_list)   { data[2] }
    # http://st-hakky.hatenablog.com/entry/2017/01/17/165137
    # エポック数はデータセットが完全に通過した回数を指します
    let(:iter_per_epoch)  { limit / batch_size }
    # 何エポックか？
    let(:epoch_size)      { iterm_num / iter_per_epoch }

    ## 明日ここから

    it { expect(train_loss_list.size).to eq iterm_num }
    it { expect(train_acc_list.size).to eq epoch_size }
    it { expect(test_acc_list.size).to eq epoch_size  }
  end

  describe '.save_params' do
    let(:path) { 'tmp/params.json' }
    let(:loaded) { Trainer.load_params(path) }
    before do
      Trainer.save_params(path, network.params)
    end
    after do
      FileUtils.rm_rf(path)
    end
    it { expect(File.exist?(path)).to eq true }
    it { expect(NP.array_equal(network.params.values[0], loaded.values[0])).to eq true }
  end
end
