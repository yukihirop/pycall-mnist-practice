FROM tnantoka/miniconda-ruby

# Sinatra
ADD ./app /opt/app
WORKDIR /opt/app

RUN bundle

RUN useradd -m myuser
USER myuser

CMD ruby app.rb -p $PORT -o 0.0.0.0
