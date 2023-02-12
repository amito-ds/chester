def round_value(x):
    try:
        if abs(x) < 10:
            return round(x, 2)
        elif abs(x) < 100:
            return round(x, 1)
        else:
            return round(x)
    except:
        return None
