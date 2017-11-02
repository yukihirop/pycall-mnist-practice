class Loader
  class << self
    def load_mnist(one_hot_label = false, limit = nil, test = true, train = true)
      # 入力x
      # テストt
      x_train, t_train = load_or_blank('train', one_hot_label, !train)
      x_test, t_test = load_or_blank('t10k', one_hot_label, !test)

      # sizeが4の配列(dataとする)
      # x_train=data[0]は784列×1(limit)行  (x_train.shape[1]=network.input_size=784)
      # t_train=data[1]は10列×1(limit)行
      # x_test=data[2]は784列×1(limit)行   (x_test.shape[1]=network.input_size=784)
      # t_test=data[3]は10列×1(limit)行
      [x_train, t_train, x_test, t_test].map { |a| limit.to_i.zero? ? a : a[0...limit] }.map { |a| NP.array(a) }
    end

    private

    def load_or_blank(type, one_hot_label, blank)
      return [[], []] if blank
      images = load_images(type)
      labels = load_labels(type)
      labels = one_hot_labels(labels) if one_hot_label
      [images, labels]
    end

    def load_images(type)
      Mnist.load_images(path("#{type}-images-idx3"))[2].map do |image|
        image.unpack('C*').map { |p| p / 255.0 }
      end
    end

    def load_labels(type)
      Mnist.load_labels(path("#{type}-labels-idx1"))
    end

    def path(file)
      File.expand_path("../../data/#{file}-ubyte.gz", __FILE__)
    end

    def one_hot_labels(x)
      x.map do |label|
        Array.new(10, 0).tap { |a| a[label] = 1 }
      end
    end

  end
end
