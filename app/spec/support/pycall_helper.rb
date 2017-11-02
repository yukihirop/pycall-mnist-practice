module PyCallHelper
  def np_array_to_a(array, digits = nil)
    Util.np_array_to_a(array).map { |v| digits.nil? ? v : v.round(digits) }
  end
end
