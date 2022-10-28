import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear(x, a, b):
    return a*x+b

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def convert_to_float(string):
    string = string.replace(",", ".")
    print(float(string))

def opening_data(data_file, variables):
    file = open(data_file, "r")
    line_count = 0
    for line in file:
        if line != "\n":
            line_count += 1
    file.close()

    meritve = line_count-3
    data = np.zeros([meritve, variables])
    variables = []
    units = []


    i=0
    with open(data_file, newline='') as f:
        reader = csv.reader(f, delimiter=' ')
        for f in reader:
            if i ==0:
                numerator = 0
                for j in range(len(f)):
                    if f[j] != '':
                        variables.append(f[j])
                        numerator +=1
            elif i == 1:
                numerator = 0
                for j in range(len(f)):
                    if f[j] != '':
                        units.append(f[j])
                        numerator +=1
            if i>1 and i<line_count-1:
                numerator = 0
                for j in range(len(f)):
                    if f[j] !='':
                        f[j] = f[j].replace(",", ".")
                        data[i-2, numerator] = f[j]
                        numerator += 1
            i=i+1
    return data, variables, units


variables = 17
name1 = 'lin_M43'
openmap1 = 'E:\\DMA\\'
data_file1 = openmap1 + name1 +'.txt'  # ime datotekbe z lokacijo

data1, variables1, units1 = opening_data(data_file1, variables)

name2 = 'lin_Z4-vlakna(kompozit)'
openmap2 = 'E:\\DMA\\'
data_file2 = openmap2 + name2 +'.txt'  # ime datotekbe z lokacijo
data2, variables2, units2 = opening_data(data_file2, variables)

labels = ["PVDF+PVP vlakna", "PVDF_HFP+PVP film"]

print(variables1)
print(units1)
data2init=np.zeros(len(data2[:, 1]))
for i in range(len(data2[:, 1])):
    data2init[i] = data2[i, 13]
plt.plot(data2[:, 1], data2[:, 13])
plt.plot(data2init)
#fittanje gaussa
a = np.min(data2[:150, 13])
b = np.max(data2[:150, 13])-np.min(data2[:150, 13])
data2[:150, 13]=(data2[:150, 13]-a)/b
plt.plot(data2init)
plt.show()
popt,pcov = curve_fit(gaus,data2[:150, 1], data2[:150, 13], p0=[1, -30, 20])
print(popt)
print(pcov)
gauss_fit = np.zeros(len(data2[:, 1]))
for i in range(len(data2[:, 1])):
    gauss_fit[i] = gaus(data2[i, 1], *popt)

gauss_fit_normal = gauss_fit*b+a
plt.plot(gauss_fit_normal)
plt.plot(gauss_fit)
plt.show()
data2[:, 13] = data2init
plt.plot(data2init)
plt.show()
fig, ax = plt.subplots(2)
ax[0].plot(data2[:, 1], data2[:, 10])
ax[1].plot(data2[:, 1], data2[:, 13])
ax[1].plot(data2[:, 1], gauss_fit)
ax[0].set_xlabel(variables1[5] + units1[5])
ax[0].set_ylabel(variables1[4] + units1[4])
ax[1].set_xlabel(variables1[5] + units1[5])
ax[1].set_ylabel(variables1[4] + units1[4])
ax[0].grid()
ax[1].grid()
plt.show()



fig, ax = plt.subplots(2)
ax[0].plot(data2[:, 1], data2[:, 10])
ax[1].plot(data2[:, 1], data2[:, 13])
ax[0].set_xlabel(variables1[1] + units1[1])
ax[0].set_ylabel(variables1[10] + units1[10])
ax[1].set_xlabel(variables1[1] + units1[1])
ax[1].set_ylabel(variables1[13] + units1[13])
ax[0].grid()
ax[1].grid()
plt.show()


#linear fit
popt1,pcov1 = curve_fit(linear,data1[:3, 5], data1[:3, 4])
popt2,pcov2 = curve_fit(linear,data2[:, 5], data2[:, 4])
print(popt1)
print(pcov1)
print(popt2)
print(pcov2)
fitlin1 = np.zeros(len(data1[:, 5]))
fitlin1[:] = linear(data1[:, 5], *popt1)

fitlin2 = np.zeros(len(data2[:, 5]))
fitlin2[:] = linear(data2[:, 5], *popt2)


fig, ax = plt.subplots(2)
plt.suptitle('Test linearnosti na filmih in vlaknih', fontsize=10)
ax[0].set_title('Film')
ax[1].set_title('Vlakna')
ax[0].errorbar(data1[:, 5], data1[:, 4], yerr=0.001, fmt='o')
ax[0].plot(data1[:, 5], fitlin1)
ax[1].errorbar(data2[:, 5], data2[:, 4], yerr=0.001, fmt='o')
ax[1].plot(data2[:, 5], fitlin2)
ax[0].set_xlabel(variables1[5] + units1[5], fontsize=10)
ax[0].set_ylabel(variables1[4] + units1[4], fontsize=10)
ax[1].set_xlabel(variables1[5] + units1[5], fontsize=10)
ax[1].set_ylabel(variables1[4] + units1[4], fontsize=10)
ax[0].grid()
ax[1].grid()
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)
plt.show()


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel(variables1[1]+units1[1])
ax1.set_ylabel(variables1[10]+units1[10], color=color)
ax1.plot(data2[:, 1], data2[:, 10], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(variables1[13]+units1[13], color=color)  # we already handled the x-label with ax1
ax2.plot(data2[:, 1], data2[:, 13], color=color)
ax2.plot(data2[:, 1], gauss_fit_normal, color='y')
ax2.tick_params(axis='y', labelcolor=color)


fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



print(f'{variables1[13]}, variables')
fig, ax = plt.subplots(2)
ax[0].plot(data1[:, 5], data1[:, 4])
ax[1].plot(data2[:, 5], data2[:, 4])
ax[0].set_xlabel(variables1[5] + units1[5])
ax[0].set_ylabel(variables1[4] + units1[4])
ax[1].set_xlabel(variables1[5] + units1[5])
ax[1].set_ylabel(variables1[4] + units1[4])
ax[0].grid()
ax[1].grid()
plt.show()

M1real_error = (10**(-5)+data1[:, 10])/data1[:, 10]-1
M2real_error = (10**(-5)+data2[:, 10])/data2[:, 10]-1
M1imaginary_error = (10**(-5)+data1[:, 11])/data1[:, 11]-1
M2imaginary_error = (10**(-5)+data2[:, 11])/data2[:, 11]-1
tandelta1_error = M1real_error+M1imaginary_error
tandelta2_error = M2real_error+M2imaginary_error

fig, ax = plt.subplots(2, 2)
plt.suptitle('Meritve kompleksne elastične konstante za nanovlakna in filme narejena iz nanokompozita', fontsize=20)
ax[0, 0].errorbar(data1[:, 1], data1[:, 10], yerr=10**(-5)) #row=0, col=0
ax[0, 0].errorbar(data2[:, 1], data2[:, 10], yerr=10**(-5))
ax[0, 0].set_xlabel(variables1[1] + units1[1], fontsize=20)
ax[0, 0].set_ylabel(variables1[10] + units1[10], fontsize=20)
ax[0, 0].set_yscale('log')
ax[1, 0].errorbar(data1[:, 1], data1[:, 11], yerr=10**(-5)) #row=1, col=0
ax[1, 0].errorbar(data2[:, 1], data2[:, 11], yerr=10**(-5)) #row=1, col=0
ax[1, 0].set_xlabel(variables1[1] + units1[1], fontsize=20)
ax[1, 0].set_ylabel(variables1[11] + units1[11], fontsize=20)
ax[1, 0].set_yscale('log')
ax[0, 1].errorbar(data1[:, 1], data1[:, 12], yerr=10**(-5)) #row=0, col=1
ax[0, 1].errorbar(data2[:, 1], data2[:, 12], yerr=10**(-5)) #row=1, col=0
ax[0, 1].set_xlabel(variables1[1] + units1[1], fontsize=20)
ax[0, 1].set_ylabel(variables1[12] + units1[12], fontsize=20)
ax[0, 1].set_yscale('log')
ax[1, 1].errorbar(data1[:, 1], data1[:, 13], yerr=tandelta1_error) #row=1, col=1
ax[1, 1].errorbar(data2[:, 1], data2[:, 13], yerr=tandelta2_error) #row=1, col=1
ax[1, 1].set_xlabel(variables1[1] + units1[1], fontsize=20)
ax[1, 1].set_ylabel(variables1[13] + units1[13], fontsize=20)
ax[0, 0].grid()
ax[0, 1].grid()
ax[1, 0].grid()
ax[1, 1].grid()
ax[0, 0].tick_params(axis='both', labelsize=20)
ax[1, 0].tick_params(axis='both', labelsize=20)
ax[0, 1].tick_params(axis='both', labelsize=20)
ax[1, 1].tick_params(axis='both', labelsize=20)
plt.legend()
plt.show()


