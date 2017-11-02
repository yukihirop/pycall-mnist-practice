require 'spec_helper'
 describe Sigmoid do
   let(:sigmoid) { Sigmoid.new }
   let(:x) { NP.array([-1.0, 1.0, 2.0]) }
   let(:forward) { sigmoid.forward(x) }

   describe '#forward' do
     subject { forward }
     it { should eq(Util.sigmoid(x)) }
   end

   describe '#backward' do
     before do
       forward
     end

     let(:dout) { 2 }
     subject { sigmoid.backward(dout) }
     it { should eq(dout * Util.sigmoid_grad(x)) }
   end
 end
