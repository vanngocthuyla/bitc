import pickle

def _extract_dg_dh_p0_ls(trace_file):
    """
    """
    data = pickle.load( open(trace_file, "r") )

    dg = data["DeltaG"]
    dh = data["DeltaH"]
    p0 = data["P0"]
    ls = data["Ls"]

    tds = dh - dg

    return (dg, dh, tds, p0, ls)