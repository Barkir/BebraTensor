def format_value(value)
    if value.is_a?(Array)
        "{#{value.join(", ")}}"
    elsif value.is_a?(Float)
        "#{value}f"
    else
        value
    end

end
