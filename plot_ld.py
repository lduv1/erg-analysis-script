import tkinter as tk
# root = tk.Tk()
# root.title("root")

import csv
import numpy as np
import sys, getopt


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

COL_TO_PLOT = 4         #column of data in sheet file to plot


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

        with open('csv_out/' + newFileName, 'w', newline='') as newCsvFile:
            outfile = csv.writer(newCsvFile, dialect='excel') 
            outfile.writerow(headers)
            for line in data:
                # print(line)
                outfile.writerow([line[0]] + [item for item in line[1]])
    
    def csv_print_both_data(self, headers, OD, OS, fileNameAddition):
        newFileName = self.filename.split('/')[-1].split('.')[-2] + '_' + fileNameAddition + ".csv"

        with open('csv_out/' + newFileName, 'w', newline='') as newCsvFile:
            outfile = csv.writer(newCsvFile, dialect='excel')

            hrow = [headers[0]]
            for header in headers[1:]:
                hrow += ['OD ' + header]
                hrow += ['OS ' + header]
            outfile.writerow(hrow)

            for lineA, lineB in zip(OD, OS):
                row = [lineA[0]]
                for i in range(int(self.headers['Waveforms'])):
                    row += [lineA[1][i]]
                    row += [lineB[1][i]]
                outfile.writerow(row)


    def parse_headers(self, csvInput):
        #ignore csvInput[0], useless header
        for item in csvInput[1]:
            key,value = item.split('=')
            self.headers[key] = value
        #ignore csvInput[2], useless header
        self.headers["Interval"] = int(csvInput[3][-1])
        self.headers["MaxTime"] = int(self.headers["DataPoints"]) * (self.headers["Interval"] * 0.001)

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

        # if log:
            # print("--LOG: OD")
            # self.log_print_data(self.dataOD)
            # print('--LOG: OS')
            # self.log_print_data(self.dataOS)

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

        

    def a_wave_fit(self, data):

        xvals = [line[0] for line in data]
        startidx = xvals.index(A_WAVE_START)
        endidx = xvals.index(A_WAVE_END)

        print(data[startidx])
        print(data[endidx])

        # for i in range(int(self.headers['Waveforms'])):
        for i in [COL_TO_PLOT]:
            yvals = [line[1][i] for line in data[startidx:endidx]]
            minval = min(yvals)
            minidx = yvals.index(minval)
            print(minidx)
            print(yvals[minidx])
            print(data[minidx + startidx])

            endcurveval = minval #* 0.8
            endcurveidx = 0
            for j in range(minidx):
                if yvals[j] >= endcurveval >= yvals[j+1]:
                    endcurveidx = j
                    print("found at " + str(j))
                    break
            print(data[startidx + endcurveidx])


            curvexvals = xvals[startidx:startidx + endcurveidx + 1]
            curveyvals = yvals[:endcurveidx + 1]

            print("")
            print(curvexvals)
            print(curveyvals)
            # print(xvals)
            # testy = a_wave_func(curvexvals,1,2)
            # print(testy)

            # print(a_wave_func(110, 1, 1))

            popt, pcov = curve_fit(a_wave_func, curvexvals, curveyvals, p0=(-500, 0, 80), bounds=([minval, 0. , 40.], [3., 20., 400]))
            # popt, pcov = curve_fit(a_wave_func, [1,2,3,4,5], [1,2,3,4,5])
            print(popt)
            plt.plot(xvals, [a_wave_func(val, *popt) for val in xvals], 'r')
            # plt.plot(xvals, [a_wave_func(val, -760, 1, 10) for val in xvals], 'r')


    def plot(self, data, color):
        xvals = [line[0] for line in data]
        yvals = [line[1][COL_TO_PLOT] for line in data]

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

        OSBase = self.get_baseline(self.dataOS)
        OSRebased = self.strip_and_rebase(self.dataOS, OSBase)
        # print(ODBase)
        # self.log_print_data(ODRebased)

        self.csv_print_data(self.dataHeaders, self.dataOD, "ODparsed")
        self.csv_print_data(self.dataHeaders, self.dataOS, "OSparsed")
        self.csv_print_data(self.dataHeaders, ODRebased, "rebased")

        self.csv_print_both_data(self.dataHeaders, self.dataOD, self.dataOS, "both")

        # self.filter_data(ODRebased)
        OSFiltered = self.filter_data(OSRebased)
        self.a_wave_fit(OSRebased)

        self.csv_print_data(self.dataHeaders,OSFiltered,"filtered")
        self.plot(OSRebased, 'b')
        self.plot(OSFiltered, 'k')
        # self.plot(self.dataOD)



        


try:
    opts, args = getopt.getopt(sys.argv[1:],"f:l:",["file=", "log="])
except getopt.GetoptError:
    print("Command Line Error")

inputfile = ""
log = 0
for opt, arg in opts:
    if opt in ("-f", "--file"):
        inputfile = arg
    elif opt in ("-l", "--log"):
        log = int(arg)

d = DataFile(inputfile) if inputfile != "" else  DataFile(read_sheets())

plt.show()
