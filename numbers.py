import numpy as np

def round_exact(f, digits):
    """Rounds `f` to the nearest `digits` behind the decimal point, and returns a string with the
    correct number of trailing zeros"""
    if np.isnan(f): return "nan"
    if digits <= 0:
        # There should be no numbers after the decimal, so why bother continuing
        return f"{int(round(f, digits))}"
    
    new_f = f"{round(f, digits)}"
    period_location = new_f.find(".")
    if period_location == -1:
        # There was no period. Add the required number of decimals.
        if digits > 0:
            new_f += "." + "0"*digits
    else:
        num_decimals = len(new_f) - period_location - 1
        if num_decimals < digits:
            new_f = new_f + "0" * (digits - num_decimals)
    return new_f

def sci_not(f, digits=None):
    """Returns `f` in scientific notation.
    # Arguments:
    - `digits`: set to None to return the fraction and exponent. Set to an integer to return a LaTeX-formatted string with the specified precision"""
    if np.isnan(f): return (np.nan, 0)
    power = int(np.floor(np.log10(np.abs(f))))
    excess = f / 10**power
    if digits is None:
        return excess, power
    else:
        return f"{round_exact(f, digits)} \\times 10^{{{power}}}"
        
def unc_latex(f, error):
    """Returns a LaTeX-formatted string for `f` with uncertainty `error`"""
    if np.isnan(f): return "nan"
    excess_f, power_f = sci_not(f)
    excess_e, power_e = sci_not(error)
    digits = power_f - power_e
    f_str = round_exact(excess_f, digits)
    e_str = round_exact(excess_e, 0)
    return f"{f_str}({e_str}) \\times 10^{{{power_f}}}"