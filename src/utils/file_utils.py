import numpy as np


class FileUtils:
    def __init__(self):
        pass

    @staticmethod
    def save_multi_dimensional_file(file_name, data):
        with file(file_name, 'w') as outfile:
            # I'm writing a header here just for the sake of readability
            # Any line starting with "#" will be ignored by numpy.loadtxt
            outfile.write('# Array shape: {0}\n'.format(data.shape))

            # Iterating through a ndimensional array produces slices along
            # the last axis. This is equivalent to data[i,:,:] in this case
            for data_slice in data:
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                np.savetxt(outfile, data_slice, fmt='%-7.2f')

                # Writing out a break to indicate different slices...
                outfile.write('# New slice\n')

    @staticmethod
    def save_single_dimenstional_file(file_name, data):
        with file(file_name, 'w') as outfile:
            np.savetxt(outfile, data, fmt='%-7.2f')
