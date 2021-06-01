import datetime
import struct 
import numpy as np

class radar_frame():

    def __init__(self):

        self.header = ''
        self.data = []
        self.capture = ''
        self.time = 0.0

    def add_data(self, new_data):
        self.data.extend(new_data)
        self.time = datetime.datetime.now()
        
    def get_data(self):
        return self.data

    def set_time(self, time):
        self.time = time

    def clear(self):
        self.data = bytearray()
        self.time = 0.0

    def disp(self):

        pass

    def convert(self):

        pass



