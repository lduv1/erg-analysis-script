import tkinter as tk
root = tk.Tk()
root.title("root")

import csv
import numpy as np
import sys, getopt, os


from scipy import signal
from scipy.optimize import curve_fit
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt


#Defined variables
BASELINE_START = 10.0   #when to start calculation of the baseline
BASELINE_END = 80.0     #when to end calculation of the baseline

STRIP_START = 5.0       #miliseconds to strip from the beginning of the data
STRIP_END = 10.0        #miliseconds to strip from the end of the data

A_WAVE_START = 105.0      #milisecond the A wave starts at
A_WAVE_END = 300.0        #milisecond the A wave ends at

COL_TO_PLOT = -1         #column of data in sheet file to plot

OUTPUT_DIRECTORY = "csv_out"

#lookup dict light_intenisty in log(cd*s/m^2)
USE_LIGHT_INTENSITY = 0
LIGHT_INTENSITY = {
    #dark adapted dim sheet
    '50mA-OD3-20us' : -4.64,
    '50mA-OD3-40us' : -3.93,
    '50mA-OD3-200us--3.3' :	-3.06,
    '50mA-OD3-400us' : -2.74,
    '50mA-OD3-800us' : -2.43,
    '50mA-OD3-3000us' : -1.85,
    #dark adapted bright sheet
    '50mA-20us--1.84' : -1.64,
    '50mA-40us' : -0.93,
    '50mA-200us--0.3' : -0.06,
    '900mA-100us-0.36' : 0.63,
    '900mA-400us' : 1.24,
    '900mA-2000us-1.71' : 1.93,
    '900mA-6000us-2.07' : 2.41,
    #light adapted
    '900mA-100us-bkg-200mA' : 0.63,
    '900mA-400us-bkg-200mA' : 1.24,
    '900mA-2000us-bkg-200mA' : 1.93,
    '900mA-6000us-bkg-200mA' : 2.41
}

PLOT_NUM = 1

def a_wave_func(x,a,b,c):
    # xx = x**2
    # bb = 17.38 * b
    # return a * x + b
    # return a * (1 - -17.38 * b * (x**2)))
    # return a * (1 - np.exp(x) * b)
    # return a * (1 - np.exp(b * x))
    # return a * np.exp(-b * x**2)

    return a * (1 - np.exp(-0.01738 * b * (x-c)**2))

#Locate sheet files
def read_sheets():

    name = askopenfilename(initialdir = "~/py3eg/", title = "Select File", filetypes=[("Sheet files","*.sheet")])

    return name

class DataFile:

    def log_print_data(self, data):
        for i in data:
            print(i)
    
    def csv_print_data(self, headers, data, fileNameAddition):
        newFileName = self.filename.split('/')[-1].split('.')[-2] + '_' + fileNameAddition + ".csv"
        # print(headers)
        # print(data)
        # print(self.filename)
        newHeaders = ['ms'] + self.get_headers(headers)

        if not os.path.exists(OUTPUT_DIRECTORY):
            os.mkdir(OUTPUT_DIRECTORY)

        with open(OUTPUT_DIRECTORY + '/' + newFileName, 'w', newline='') as newCsvFile:
            outfile = csv.writer(newCsvFile, dialect='excel') 
            outfile.writerow(newHeaders)
            for line in data:
                # print(line)
                outfile.writerow([line[0]] + [item for item in line[1]])
        if log:
            print("--LOG: wrote file " + newFileName)
    
    def csv_print_both_data(self, headers, OD, OS, fileNameAddition):
        newFileName = self.filename.split('/')[-1].split('.')[-2] + '_' + fileNameAddition + ".csv"
        newHeaders = ['ms'] + self.get_headers(headers)

        if not os.path.exists(OUTPUT_DIRECTORY):
            os.mkdir(OUTPUT_DIRECTORY)

        with open(OUTPUT_DIRECTORY +'/' + newFileName, 'w', newline='') as newCsvFile:
            outfile = csv.writer(newCsvFile, dialect='excel')

            hrow = [newHeaders[0]]
            for header in newHeaders[1:]:
                hrow += ['OD ' + header]
                hrow += ['OS ' + header]
            outfile.writerow(hrow)

            for lineA, lineB in zip(OD, OS):
                row = [lineA[0]]
                for i in range(int(self.headers['Waveforms'])):
                    row += [lineA[1][i]]
                    row += [lineB[1][i]]
                outfile.writerow(row)
        if log:
            print("--LOG: wrote file " + newFileName)


    def parse_headers(self, csvInput):
        #ignore csvInput[0], useless header
        for item in csvInput[1]:
            key,value = item.split('=')
            self.headers[key] = value
        #ignore csvInput[2], useless header
        self.headers["Interval"] = int(csvInput[3][-1])
        self.headers["MaxTime"] = int(self.headers["DataPoints"]) * (self.headers["Interval"] * 0.001)
    
    def get_headers(self, headers):
        if USE_LIGHT_INTENSITY == 1:
            return [str(LIGHT_INTENSITY[header]) for header in headers[1:]]
        
        return [header for header in headers[1:]]
    
    def get_header(self, header):
        if USE_LIGHT_INTENSITY == 1:
            return str(LIGHT_INTENSITY[header])
        
        return header

    def parse_data(self, csvInput):
        ODStart, ODEnd = 5, 6 + int(self.headers['DataPoints'])
        OSStart, OSEnd = ODEnd + 1, ODEnd + 2 + int(self.headers['DataPoints'])

        #strip unnessessary columns
        self.dataHeaders += csvInput[ODStart][1:-2]
        for line in csvInput[ODStart+1:ODEnd]:
            time = [float(line[1])]
            nums = [float(num) for num in line[2:-2]]
            self.dataOD += [time + [nums]]
            #strip first and last 2 columns, convert to numbers

        for line in csvInput[OSStart+1:OSEnd]:
            time = [float(line[1])]
            nums = [float(num) for num in line[2:-2]]
            self.dataOS += [time + [nums]]
            #strip first and last 2 columns, convert to numbers

        if log > 1:
            print("--LOG: OD")
            self.log_print_data(self.dataOD)
            print('--LOG: OS')
            self.log_print_data(self.dataOS)

    def get_baseline(self, data):
        sum = [0] * int(self.headers['Waveforms'])
        start_index = int(BASELINE_START / (self.headers["Interval"] * 0.001))
        end_index = int(BASELINE_END / (self.headers["Interval"] * 0.001))
        for line in data[start_index:end_index]:
            for i in range(int(self.headers['Waveforms'])):
                sum[i] += line[1][i] 
        return [round(s/(end_index-start_index),3) for s in sum]

    def strip_and_rebase(self, data, baseline):
        new_data = []
        for line in data:
            if STRIP_START < line[0] < self.headers["MaxTime"] - STRIP_END:
                rebased_line = [line[0]] + [[round(line[1][i] - baseline[i],3) for i in range(int(self.headers['Waveforms']))]]
                # rebased_line = [line[0]] + [[round(x - baseline,3) for x in line[1]]] + [round(line[1][i] - baseline,3)]
                new_data += [rebased_line]
                # print(line)
                # print(rebased_line)

        return new_data

    def filter_data(self, data):
        newdata = []
        #copy time intervals
        for line in data:
            newdata += [[line[0], []]]

        # self.log_print_data(newdata)
        b, a = signal.butter(3, [0.03, 0.3], 'bandpass', False)

        for i in range(int(self.headers['Waveforms'])):
            yvals = [line[1][i] for line in data]

            # b, a = signal.cheby2(5, 30, [0.03, 0.3], 'bandpass', False)
            # zi = signal.lfilter_zi(b, a)
            # z, _ = signal.lfilter(b, a, yvals, zi=zi*yvals[0])
            # z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
            newyvals = signal.filtfilt(b, a, yvals)
            for j in range(len(newyvals)):
                newdata[j][1] += [round(newyvals[j],3)]
        
        return newdata

        

    def a_wave_fit(self, data, col):

        xvals = [line[0] for line in data]
        startidx = xvals.index(A_WAVE_START)
        endidx = xvals.index(A_WAVE_END)

        if log:
            print("--LOG: Starting data for A-wave fit - " + str(data[startidx]))
            print("--LOG: Ending data for A-wave fit - " + str(data[endidx]))

        # for i in range(int(self.headers['Waveforms'])):
        if col >= int(self.headers['Waveforms']):
            print("--ERROR: COL_TO_PLOT is larger than the number of columns")
            quit()
        for i in [col]: #if only doing fitting for plotting

            yvals = [line[1][i] for line in data[startidx:endidx]]
            minval = min(yvals)
            minidx = yvals.index(minval)
            if log:
                print("--LOG: minimum value found at index" + str(minidx) + " values: " + str(yvals[minidx]))
                print("--LOG: Data between start and minimum" + str(data[minidx + startidx]))

            endcurveval = minval * 0.8
            endcurveidx = 0
            for j in range(minidx):
                if yvals[j] >= endcurveval >= yvals[j+1]:
                    endcurveidx = j
                    if log:
                        print("--LOG: minimum * 0.8 found at index: " + str(j))
                    break
            # print(data[startidx + endcurveidx])

            curvexvals = xvals[startidx:startidx + endcurveidx + 1]
            curveyvals = yvals[:endcurveidx + 1]

            if log:
                print("")
                print("--LOG: x vals used for curve fitting: " + str(curvexvals))
                print("--LOG: y vals used for curve fitting: " + str(curveyvals))
            # testy = a_wave_func(curvexvals,1,2)
            # print(testy)

            # print(a_wave_func(110, 1, 1))

            popt, pcov = curve_fit(a_wave_func, curvexvals, curveyvals, p0=(minval*2, 0, 80), bounds=([minval*2, 0. , 40.], [3., 20., 400]))
            # popt, pcov = curve_fit(a_wave_func, [1,2,3,4,5], [1,2,3,4,5])
            if log:
                print("--LOG: values used for A-wave curve function" + str(popt))
            plt.plot(xvals[startidx:], [a_wave_func(val, *popt) for val in xvals[startidx:]], 'r')
            # plt.plot(xvals, [a_wave_func(val, -760, 1, 10) for val in xvals], 'r')


    def plot(self, data, color, col):
        if col >= int(self.headers['Waveforms']):
            print("--ERROR: col is larger than the number of columns")
            quit()

        xvals = [line[0] for line in data]
        yvals = [line[1][col] for line in data]

        # basexvals = [line[0] for line in self.dataOD]
        # baseyvals = [line[-1] for line in self.dataOD]

        plt.plot(xvals, yvals, color)
        # plt.plot(basexvals, baseyvals,'g')
        # plt.plot(xvals, z, 'r--')
        # plt.plot(xvals, z2, 'r') 
        # plt.plot(xvals, y, 'k')
        # plt.plot(xvals, yvals-y, 'r')



    #runs on start
    def __init__(self, filename):
        self.dataOD = []
        self.dataOS = []
        self.dataHeaders = []
        self.headers = {}
        self.fignum = 1

        if filename == "":
            print("File Does not exist")
            quit()

        if log:    
            print("--LOG: Openning " + filename)
        self.filename = filename

        csvInput = list(csv.reader(open(filename), delimiter='\t'))
        self.parse_headers(csvInput)
        self.parse_data(csvInput)

        ODBase = self.get_baseline(self.dataOD)
        ODRebased = self.strip_and_rebase(self.dataOD, ODBase)
        ODFiltered = self.filter_data(ODRebased)

        OSBase = self.get_baseline(self.dataOS)
        OSRebased = self.strip_and_rebase(self.dataOS, OSBase)
        OSFiltered = self.filter_data(OSRebased)

        
        self.csv_print_data(self.dataHeaders, self.dataOD, "ODparsed")
        self.csv_print_data(self.dataHeaders, self.dataOS, "OSparsed")
        self.csv_print_data(self.dataHeaders, ODRebased, "rebased")
        self.csv_print_data(self.dataHeaders, OSFiltered,"filtered")
        self.csv_print_both_data(self.dataHeaders, self.dataOD, self.dataOS, "both")
        self.csv_print_both_data(self.dataHeaders, ODRebased, OSRebased, "both_rebased")
        self.csv_print_both_data(self.dataHeaders, ODFiltered, OSFiltered, "both_filtered")

        cols = []
        if COL_TO_PLOT >= 0: #plot one column
            cols = [COL_TO_PLOT]
        else : #plot all columns
            cols = range(int(self.headers['Waveforms']))

        for i in cols:
            plt.figure(self.fignum)
            self.fignum += 1
            plt.suptitle("OD " + self.get_header(self.dataHeaders[i+1]))
            self.plot(ODRebased, 'b', i)
            self.plot(ODFiltered, 'k', i)
            self.a_wave_fit(ODRebased, i)

            plt.figure(self.fignum)
            self.fignum += 1
            plt.suptitle("OS " + self.get_header(self.dataHeaders[i+1]))
            self.plot(OSRebased, 'b', i)
            self.plot(OSFiltered, 'k', i)
            self.a_wave_fit(OSRebased, i)

def usage():
    print("-h, --help: print help")
    print("-f, --file [FILENAME]: define the file to use")
    print("-l, --log [LOGLEVEL]: print logs up to the log level")
    print("-c, --col [COLNUMBER]: plot only a single column")
    print("-li, --light: use the light intensity conversion for column headers")


try:
    opts, args = getopt.getopt(sys.argv[1:],"h:f:l:c:li",["help","file=", "log=", "col=","light"])
except getopt.GetoptError:
    usage()
    sys.exit()


inputfile = ""
log = 0
for opt, arg in opts:
    if opt in ("-h", "--help"):
        usage()
        sys.exit()
    elif opt in ("-f", "--file"):
        inputfile = arg
    elif opt in ("-l", "--log"):
        log = int(arg)
    elif opt in ("-c", "--col"):
        COL_TO_PLOT = int(arg)
    elif opt in ("-li", "--light"):
        USE_LIGHT_INTENSITY = 1



if inputfile == "":
    inputfile = read_sheets()

d = DataFile(inputfile)
root.update()

plt.show()
