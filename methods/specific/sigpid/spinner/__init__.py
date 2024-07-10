import sys
import threading
import time

class Spinner:
    """https://stackoverflow.com/a/39504463/3727678"""
    busy = False
    delay = 0.1

    def spinning_cursor(self):
        while 1:
            for cursor in ['|', '/', '-', '\\']:
                yield self.message + ' ' + cursor
                #yield cursor + self.message

    def __init__(self, message='', delay=0.2):
        self.message = message
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay):
            self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b' * (len(self.message) + 2))
            sys.stdout.flush()
        sys.stdout.write(self.message + '. Done!\n')

    def start(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        self.busy = False
        time.sleep(self.delay)

#Example
if __name__ == '__main__':
    s = Spinner('Loading')
    s.start()
    #Long Operation
    time.sleep(10)
    s.stop()
