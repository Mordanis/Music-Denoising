import sys
import time

#for i in range(10):
#    sys.stdout.write(str(i))
#    time.sleep(.1)
    #sys.stdout.flush()

for i in range(10):
    sys.stdout.write('\r\r{}'.format(i))
    sys.stdout.flush()
    time.sleep(.1)
    
