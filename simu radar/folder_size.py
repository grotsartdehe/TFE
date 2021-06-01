import os

def f_size(folder):
'''
	Returns the size of the folder
'''

        folder_size = 0
        for(path, dirs, files) in os.walk(folder):
                for file in files:
                        filename = os.path.join(path, file)
                        folder_size += os.path.getsize(filename)
                        

        return folder_size


if __name__ == '__main__':

        path = '/home/jetson-nano-tfe/Desktop/tfe_codes/cam_and_radar/data'
        s = f_size(path)
        print('Size : ', s)
