class NP
  class << self
    def size(object, axis)
      Numpy.size(object, axis)
    end


    def array(object)
      Numpy.array(object)
    end

    def array_equal(a1, a2)
      Numpy.array_equal(a1, a2)
    end

    def dot(a, b)
      Numpy.dot(a, b)
    end

    def copy(a)
      Numpy.copy(a)
    end

    def zeros_like(a)
      Numpy.zeros_like(a)
    end

    # axis = 0 列でたす
    # axis = 1 行で足す
    def sum(a, axis = nil)
      axis.nil? ? Numpy.sum(a) : Numpy.sum(a, axis)
    end

    def exp(x)
      Numpy.exp(x)
    end

    # 配列の最大要素のindexを返す
    # axis=0 列
    # axis=1 行
    def argmax(a, axis)
      Numpy.argmax(a, axis)
    end

    def max(a, axis)
      Numpy.max(a, axis)
    end

    # 正規分布
    def randn(d0, d1)
      Numpy.random.randn(d0, d1)
    end

    # a以下の数字でsizeのNumpy.ndarray
    def choice(a, size)
      Numpy.random.choice(a, size)
    end

    # 0.の配列
    def zeros(shape)
      Numpy.zeros(shape)
    end

    #
    def log(x)
      Numpy.log(x)
    end

    def arange(stop)
      Numpy.arange(stop)
    end

    # イテレータオブジェクトを返す
    def nditer(op, flags, op_flags)
      Numpy.nditer(op, flags, op_flags)
    end
  end
end
