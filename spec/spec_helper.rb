require 'simplecov'
SimpleCov.start

require './lib'

require 'fileutils'

require './spec/support/pycall_helper'

RSpec.configure do |config|
  config.include PyCallHelper
end
