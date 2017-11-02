require 'bundler'
Bundler.require

require 'json'

include PyCall::Import

require './lib/loader'
require './lib/util'
require './lib/trainer'
require './lib/predictor'
require './lib/np'
require './lib/sgd'
require './lib/sigmoid'
require './lib/affine'
require './lib/softmax_with_loss'
require './lib/relu'
