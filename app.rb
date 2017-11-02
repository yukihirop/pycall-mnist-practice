require './lib'

require 'sinatra'

predicator = Predicator.new('data/params.json')

get '/' do
  slim :index
end

post '/predict' do
  canvas, pixels = Predicator.parse(params[:data_url])
  y = predicator.predict(pixels)
  label = y.argmax.().to_s.to_i
  percent(y[label]*100).round(2)
  # canvas.to_data_urlは実際に手書きしたへのurl???
  { label: label, image: canvas.to_data_url, percent: percent }.to_json
end
