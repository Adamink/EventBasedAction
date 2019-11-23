import time
def get_time():
    return time.time()

def fmt_elapsed_time(e):
    e = (int)(e)
    hours = e // 3600
    temp = e - 3600 * hours
    minutes = temp // 60
    seconds = temp - 60 * minutes
    s = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return s

def get_fmt_time():
    return time.strftime('%H:%M')

if __name__=='__main__':
    a = get_time()
    for i in range((int)(1e8)):
        pass
    b = get_time()
    print(fmt_elapsed_time(b - a))
