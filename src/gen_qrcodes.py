import random
import sys

import qrcode

RANDM_RANGE = 1e20 + 1

def main():
    pass

def gen_sequential_qr(num, dir):
    for i in range(num):
        msg = "%020d" % i
        f_name = "%s/%s.png" % (dir, msg)
        gen_qrcode(msg, f_name)
        print(msg) 

def gen_random_qr(num, dir, seed, random_range=RANDM_RANGE):
    random.seed(seed)
    for i in range(num):
        msg = "%020d" % random.randrange(random_range)
        f_name = "%s/%s.png" % (dir, msg)
        gen_qrcode(msg, f_name)
        print(msg) 

def gen_qrcode(msg, f_name):
    img = qrcode.make(msg)
    img.save(f_name)

if __name__ == "__main__":
    main()