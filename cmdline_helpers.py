import os

def colon_separated_range(x):
    lower = upper = None
    if x != '':
        if ':' not in x:
            print('Invalid range format; should be float:float')
            return
        l, _, h = x.partition(':')
        try:
            lower = float(l)
            upper = float(h)
        except ValueError:
            print('Invalid range format; should be float:float')
            return

    return lower, upper


def ensure_containing_dir_exists(path):
    dir_name = os.path.dirname(path)
    if dir_name != '' and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return path
