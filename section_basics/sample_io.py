import sys
import struct

with open(sys.argv[1], 'rb') as f:
    # read header
    for i in range(10): # assume the number of header lines is 10
        line = f.readline()
        print (line)
        if b'vertex ' in line:
            vnum = int(line.split(b' ')[-1]) # num of vertices
        if b'face ' in line:
            fnum = int(line.split(b' ')[-1]) # num of faces

    # read vertices
    for i in range(vnum):
        for j in range(3):
            print (struct.unpack('f', f.read(4))[0], end=' ')
        print ("")

    # read faces
    for i in range(fnum):
        n = struct.unpack('B', f.read(1))[0]
        for j in range(n):
            print (struct.unpack('i', f.read(4))[0], end=' ')
        print ("")
