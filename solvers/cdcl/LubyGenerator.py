import math

l = []
mult = 1
minu = 0

def get_next_luby_number():
    """
    Method to get the next luby number

    Parameters:
        None

    Return:
        the next Luby number in the sequence
    """
    global l
    global mult
    global minu

    size = len(l)

    to_fill = size + 1

    if math.log(to_fill + 1, 2).is_integer():
        l.append(mult)
        mult *= 2
        minu = size + 1
    else:
        l.append(l[to_fill - minu - 1])

    return l[size]


def reset_luby():
    """
    Method to reset the Luby Generator
    to initial conditions.

    Parameters:
        None

    Return:
        None
    """
    global l
    global mult
    global minu

    l = []
    mult = 1
    minu = 0