import matplotlib.pyplot as plt
import csv
from scipy import optimize
import numpy as np
import re
from scipy.signal.signaltools import wiener
import pandas as pd


class raman:

    podatki = 1024

    def __init__(self, x, y, podatki, izbrisraileigha, glaj, VSP, SP, MP, WP, VWP, Sh, mera_za_shoulderje, mera_za_dvojne_shoulderje, winer):   # podatki za v class
        self.x = x
        self.y = y
        self.podatki = podatki
        self.izbrisraileigha = izbrisraileigha
        self.glaj = glaj
        self.VSP = VSP
        self.SP = SP
        self.MP = MP
        self.WP = WP
        self.VWP = VWP
        self.Sh = Sh
        self.mera_za_shoulderje = mera_za_shoulderje
        self.mera_za_dvojne_shoulderje = mera_za_dvojne_shoulderje
        self.winer = winer


    @staticmethod
    def gaussian(x, amp, cen, wid):   # gaussovka za premik rayleigha
        return amp * np.exp(-(x - cen) ** 2 / wid)

    def premikrayleigha(self):    # zamakne rayleigha na 0 s fitanjem gaussovke
        a = find_nearest(self.x, -20)
        b = find_nearest(self.x, 40)
        valst = np.zeros(b - a)
        intenziteta = np.zeros(b - a)
        for i in range(b - a):
            valst[i] = self.y[a + i]
            intenziteta[i] = self.x[a + i]
        params, params_covariance = optimize.curve_fit(raman.gaussian, intenziteta, valst,
                                                       p0=([2 * 10 ** 4, 10, 24]))
        self.x = self.x - params[1]

    def gama(self):   # vecinoma pobrise gama vrhove
        for i in range(self.podatki - 2):
            if self.y[i] / (self.y[i + 1] + 0.01) > 1.04 and self.y[i] / (self.y[i - 1] + 0.01) > 1.04:
                self.y[i] = (self.y[i - 1] + self.y[i + 1]) / 2

    def pobris_raileigha(self):   # izbrise prvih nekaj podatkov ki so ponavadi raileigh in pa zelo majhni spektri
        self.y = self.y[self.izbrisraileigha:]
        self.x = self.x[self.izbrisraileigha:]

    @staticmethod
    def kubicna(x, a, b, c, d, e, f):  # uporabno za naprej za fitanje ozadja
        return f * x ** 5 + e * x ** 4 + a * x ** 3 + b * x ** 2 + c * x + d

    def fitozad(self):   # fita kubicno funkcijo in odsteje za lazjo zaznavo vrhov
        params, paramscovariance = optimize.curve_fit(raman.kubicna, self.x, self.y, p0=None)
        for i in range(len(self.y)):
            self.y[i] = self.y[i] - raman.kubicna(self.x[i], *params)

    def normalizacija(self):  # spravi podatke med 0 in 1
        self.y = self.y - np.amin(self.y)
        self.y = self.y / np.amax(self.y)

    def glajenje(self):  # zgladi podatke s tem da jih po vec sesteje skupaj in povpreci
        for i in range(int(self.glaj/2+1/2), int(len(self.y) - (self.glaj/2+1/2))):
            stevec = 0
            for j in range(self.glaj):
                stevec = stevec + self.y[int(i + j - self.glaj/2+1/2)]
            self.y[i] = stevec / self.glaj
        return self.y


    def wiener(self):

        self.y = wiener(self.y, self.winer)

    @staticmethod
    def pisanjepeak(letter, vrhovix, vrhoviy, truevrhovilist, truevrhovix, truevrhoviy, i): # zapise kje so peaksi
        truevrhovilist.append(str(round(vrhovix[i]))+letter)
        truevrhovix = np.append(truevrhovix, vrhovix[i])
        truevrhoviy = np.append(truevrhoviy, vrhoviy[i])
        return truevrhovilist, truevrhovix, truevrhoviy

    def vrhovi_in_doline(self):
        vrhovix = np.zeros([])
        vrhoviy = np.zeros([])
        dolinex = np.zeros([])
        doliney = np.zeros([])
        polozaj = np.zeros([])

        for i in range(1, len(self.x)-1):  # zapises vse vrhovi in doline ter njihove indekse
            if self.y[i-1] < self.y[i] > self.y[i+1]:
                polozaj = np.append(polozaj, i)
                vrhoviy = np.append(vrhoviy, self.y[i])
                vrhovix = np.append(vrhovix, self.x[i])
            if self.y[i-1] > self.y[i] < self.y[i+1]:
                dolinex = np.append(dolinex, self.x[i])
                doliney = np.append(doliney, self.y[i])

        checkingarray = np.zeros([len(vrhovix), 6])   # arrayi za čekirat kolk je nad bližnjimi minimumi
        checkingarray[0, :] = (doliney[3], doliney[2], doliney[1], doliney[1], doliney[2], doliney[3])  # zacetke in konce je treba bruteforce napisat
        checkingarray[1, :] = (doliney[2], doliney[1], doliney[1], doliney[2], doliney[3], doliney[4])
        checkingarray[2, :] = (doliney[1], doliney[1], doliney[2], doliney[3], doliney[4], doliney[5])
        checkingarray[len(vrhovix) - 1, :] = (doliney[len(vrhovix)-3], doliney[len(vrhovix)-2], doliney[len(vrhovix)-1], doliney[len(vrhovix)-1], doliney[len(vrhovix)-2], doliney[len(vrhovix)-3])
        checkingarray[len(vrhovix) - 2, :] = (
        dolinex[len(vrhovix) - 2], doliney[len(vrhovix) - 1], doliney[len(vrhovix) - 1], doliney[len(vrhovix) - 2],
        dolinex[len(vrhovix) - 3], dolinex[len(vrhovix) - 4])
        checkingarray[len(vrhovix) - 3, :] = (
        dolinex[len(vrhovix) - 1], dolinex[len(vrhovix) - 1], dolinex[len(vrhovix) - 2], dolinex[len(vrhovix) - 3],
        dolinex[len(vrhovix) - 4], dolinex[len(vrhovix) - 5])


        okolica = 3
        for i in range(okolica, len(vrhovix)-okolica):  # napises vse po 3 nicle na vsako stran
            for j in range(-okolica, okolica):
                checkingarray[i, j] = doliney[i+j]


        truevrhovilist = []
        truevrhovix = np.zeros([])
        truevrhoviy = np.zeros([])
        for i in range(len(vrhovix)-okolica+1):  # zapise vrhove na prblizn glede na to kolk je nad najnizjo niclo v okolici
            if vrhoviy[i]-np.min(checkingarray[i, :okolica]) > self.VSP and vrhoviy[i]-np.min(checkingarray[i, okolica:]) > self.VSP:
                truevrhovilist, truevrhovix, truevrhoviy = raman.pisanjepeak('VS_P', vrhovix, vrhoviy, truevrhovilist, truevrhovix, truevrhoviy, i)
            elif vrhoviy[i] - np.min(checkingarray[i, :okolica]) > self.SP and vrhoviy[i] - np.min(checkingarray[i, okolica:]) > self.SP:
                truevrhovilist, truevrhovix, truevrhoviy = raman.pisanjepeak('S_P', vrhovix, vrhoviy, truevrhovilist,
                                                                             truevrhovix, truevrhoviy, i)
            elif vrhoviy[i] - np.min(checkingarray[i, :okolica]) > self.MP and vrhoviy[i] - np.min(checkingarray[i, okolica:]) > self.MP:
                truevrhovilist, truevrhovix, truevrhoviy = raman.pisanjepeak('M_P', vrhovix, vrhoviy, truevrhovilist,
                                                                             truevrhovix, truevrhoviy, i)
            elif vrhoviy[i] - np.min(checkingarray[i, :okolica]) > self.WP and vrhoviy[i] - np.min(checkingarray[i, okolica:]) > self.WP:
                truevrhovilist, truevrhovix, truevrhoviy = raman.pisanjepeak('W_P', vrhovix, vrhoviy, truevrhovilist,
                                                                             truevrhovix, truevrhoviy, i)
            elif vrhoviy[i] - np.min(checkingarray[i, :okolica]) > self.VWP and vrhoviy[i] - np.min(checkingarray[i, okolica:]) > self.VWP:
                truevrhovilist, truevrhovix, truevrhoviy = raman.pisanjepeak('VW_P', vrhovix, vrhoviy, truevrhovilist,
                                                                             truevrhovix, truevrhoviy, i)
            elif vrhoviy[i] - np.min(checkingarray[i, :okolica]) > self.Sh:
                truevrhovilist, truevrhovix, truevrhoviy = raman.pisanjepeak('Sh', vrhovix, vrhoviy, truevrhovilist,
                                                                             truevrhovix, truevrhoviy, i)
            elif vrhoviy[i] - np.min(checkingarray[i, okolica:]) > self.Sh:
                truevrhovilist, truevrhovix, truevrhoviy = raman.pisanjepeak('Sh', vrhovix, vrhoviy, truevrhovilist,
                                                                            truevrhovix, truevrhoviy, i)

        spreminjanjevrh = np.array([])
        for i in range(1, len(truevrhovix)):
            spreminjanjevrh = np.append(spreminjanjevrh, find_nearest(vrhovix, truevrhovix[i]))  # zapise indekse od oznacenih vrhov v vseh vrhovih

        spreminjanjedoline = np.zeros((len(spreminjanjevrh), 5))  # ustvars array za pisanje okolice vrha (1 vrh in 1 dolina v vsako stran)
        for i in range(len(spreminjanjevrh)):   # zapises okolico vrha
            najblizja_dolina = dolinex[find_nearest(dolinex, vrhovix[int(spreminjanjevrh[i])])]  # rabs zato da povse a je to najblizja leva al najblizja desna dolina
            if vrhovix[int(spreminjanjevrh[i])] > najblizja_dolina:
                spreminjanjedoline[i, 0] = vrhoviy[int(spreminjanjevrh[i])-1]
                spreminjanjedoline[i, 1] = doliney[find_nearest(dolinex, vrhovix[int(spreminjanjevrh[i])])]
                spreminjanjedoline[i, 2] = vrhoviy[int(spreminjanjevrh[i])]
                spreminjanjedoline[i, 3] = doliney[find_nearest(dolinex, vrhovix[int(spreminjanjevrh[i])])+1]
                spreminjanjedoline[i, 4] = vrhoviy[int(spreminjanjevrh[i])+1]
            else:
                spreminjanjedoline[i, 0] = vrhoviy[int(spreminjanjevrh[i])-1]
                spreminjanjedoline[i, 1] = doliney[find_nearest(dolinex, vrhovix[int(spreminjanjevrh[i])]) - 1]
                spreminjanjedoline[i, 2] = vrhoviy[int(spreminjanjevrh[i])]
                spreminjanjedoline[i, 3] = doliney[find_nearest(dolinex, vrhovix[int(spreminjanjevrh[i])])]
                spreminjanjedoline[i, 4] = vrhoviy[int(spreminjanjevrh[i])+1]


        for i in range(np.shape(spreminjanjedoline)[0]):  # spremeni dolocene peakse k jih prej ni najdu da so soulderji v shoulderje
            if spreminjanjedoline[i, 0] > spreminjanjedoline[i, 2]+self.mera_za_shoulderje < spreminjanjedoline[i, 4]:
                # ce je med dvema peakoma en manjsi rece da ga v bistvu ni
                truevrhovilist[i] = str(round(truevrhovix[i + 1])) + "Brisi"
            elif spreminjanjedoline[i, 0]-spreminjanjedoline[i, 2] > self.mera_za_shoulderje and spreminjanjedoline[i, 2]-spreminjanjedoline[i, 1] < self.mera_za_shoulderje:
                # gleda ce je tolk manjsi od prejsnjega vrha in hkrati umes in ful velike doline pa se da je pol naprej precej nizji minimum potem je to shoulder v desno
                truevrhovilist[i] = str(round(truevrhovix[i+1]))+"Sh"
            elif spreminjanjedoline[i, 4] - spreminjanjedoline[i, 2] > self.mera_za_shoulderje and spreminjanjedoline[i, 2]-spreminjanjedoline[i, 3] < self.mera_za_shoulderje:
                # isto samo v levo
                truevrhovilist[i] = str(round(truevrhovix[i+1]))+"Sh"

        listaimen = []
        for i in range(len(truevrhovilist)-1):  # loci listo z vrhovi in valovnim stevilom na ime vrha npr. VS  in pa valovno stevilo
            match = re.match(r"([0-9]+)([a-z]+)", str(truevrhovilist[i]), re.I)
            listaimen.append(match.groups()[1])

        st_zaporednih_vrhov = np.array([])  # naredi array ki bo povedal koliko zaporednih vrhov se drzi skupaj
        stevec = 0
        for i in range(len(spreminjanjevrh)-1):  # zapises vse da ves kolk ta mocnih vrhov se drzi skp
            if spreminjanjevrh[i+1]-spreminjanjevrh[i] == 1:
                stevec = stevec+1
                if i == len(spreminjanjevrh)-2:
                    st_zaporednih_vrhov = np.append(st_zaporednih_vrhov, stevec+1)
            else:
                st_zaporednih_vrhov = np.append(st_zaporednih_vrhov, stevec+1)
                stevec = 0

        kateri_sklop_vrhov = np.zeros(len(spreminjanjevrh))  # array k pove kater vrh pripada katermu sklopu vrhov
        counter = 1
        for i in range(len(truevrhovix)-1):
            if i < np.sum(st_zaporednih_vrhov[:counter])-1:
                kateri_sklop_vrhov[i] = counter
            else:
                kateri_sklop_vrhov[i] = counter
                counter = counter + 1
        counter = 0  # to vse skp pove, da ce mas 1 shoulder pa je soseden rahlo visji ali pa manjsi je potem to tut shoulder
        for i in range(len(st_zaporednih_vrhov)):
            for j in range(int(st_zaporednih_vrhov[i])):
                if j == 0:
                    if listaimen[counter] == "Sh" and truevrhoviy[counter + 1] - truevrhoviy[counter] < self.mera_za_dvojne_shoulderje and kateri_sklop_vrhov[counter+1]-kateri_sklop_vrhov[counter] == 0:
                        print(truevrhovilist[counter])
                        truevrhovilist[counter] = str(round(truevrhovix[counter + 1])) + "Sh"
                        print("1", truevrhovix[counter + 1], (kateri_sklop_vrhov[counter]))
                elif j < int(st_zaporednih_vrhov[i]-1):
                    if listaimen[counter] == "Sh" and truevrhoviy[counter] - truevrhoviy[counter-1] < self.mera_za_dvojne_shoulderje and kateri_sklop_vrhov[counter+1]-kateri_sklop_vrhov[counter] == 0:
                        print(truevrhovilist[counter+1])
                        truevrhovilist[counter + 1] = str(round(truevrhovix[counter + 2])) + "Sh"
                        print("2", truevrhovix[counter+2], (kateri_sklop_vrhov[counter]))
                    elif listaimen[counter] == "Sh" and truevrhoviy[counter - 1] - truevrhoviy[counter] < self.mera_za_dvojne_shoulderje and kateri_sklop_vrhov[counter-2]-kateri_sklop_vrhov[counter-1] == 0:
                        print(truevrhovilist[counter-1])
                        truevrhovilist[counter - 1] = str(round(truevrhovix[counter])) + "Sh"
                        print("3", truevrhovix[counter], (kateri_sklop_vrhov[counter]))
                elif j == int(st_zaporednih_vrhov[i]-1) and counter < len(truevrhovix)-2:
                    if listaimen[counter] == "Sh" and truevrhoviy[counter - 1] - truevrhoviy[counter] < self.mera_za_dvojne_shoulderje and kateri_sklop_vrhov[counter-2]-kateri_sklop_vrhov[counter-1] == 0:
                        print(truevrhovilist[counter])
                        truevrhovilist[counter] = str(round(truevrhovix[counter+1])) + "Sh"
                        print("4", truevrhovix[counter+1], (kateri_sklop_vrhov[counter]))
                counter = counter + 1

        for i in range(len(truevrhovix)): # to pa pove une k so bli prej za zbrisat da jih dejansko zbrise
            if listaimen[i-2] == "Brisi":
                print("yay", truevrhovix[i])
                truevrhovix = np.delete(truevrhovix, i-1)
                truevrhoviy = np.delete(truevrhoviy, i-1)
                truevrhovilist.pop(i-2)


        moc_peaka = np.zeros(len(truevrhovix))
        for i in range(len(truevrhovix)):
            closest = find_nearest(vrhovix, truevrhovix[i])
            moc_peaka[i] = vrhoviy[closest]-np.min(checkingarray[closest,:])

        intenziteta = np.zeros(len(truevrhovix)-1)
        for i in range(1, len(truevrhovix)):
            intenziteta[i-1] = (moc_peaka[i]/np.max(moc_peaka[1:]))*100

        return truevrhovilist, truevrhovix, truevrhoviy, intenziteta, dolinex, doliney

def find_nearest(array, value):  # najde indeks najblizjega elementa v arrayu temu kar ti vneses
    idx = (np.abs(array - value)).argmin()
    return idx

def odpiranje_podatkov(ime, podatki):
    intenzitetakompozit = np.zeros(podatki)
    valstkompozit = np.zeros(podatki)
    # odpre podatke
    with open(ime, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        i = 0
        for f in reader:
            for j in range(2):
                if j == 0:
                    valstkompozit[i] = f[j]
                if j == 1:
                    intenzitetakompozit[i] = f[j]
            i = i + 1

    return valstkompozit, intenzitetakompozit

def zbrisanje(vrhovix, vrhoviy, listx): # zbrises vrhove k jih je pomotome dodau
    lstizbris = []
    a = 1
    print("katere vrhove hoces zbrisat, ko napises zadnjega napis se OK:")
    while a != "OK":
        a = input()
        if a != "OK":
            lstizbris.append(float(a))
    for i in range(len(lstizbris)):
        try:
            for j in range(len(vrhovix)):
                print(lstizbris[i])
                print(round(vrhovix[j]))
                if round(lstizbris[i]) == round(vrhovix[j]):
                    print("vrh je bil tam in je izbrisan")
                    vrhovix = np.delete(vrhovix, j)
                    vrhoviy = np.delete(vrhoviy, j)
                    listx.pop(j - 1)
        except:
            pass
    print(listx)
    return vrhovix, vrhoviy, listx


def dodajanje(x, y, vrhovix, vrhoviy, listx, intenziteta, dolinex, doliney, VWP, WP, MP, SP, VSP, ime_grafa): #dodas nove vrhove k jih ni vidu
    lstdodaj = []
    a = 1
    print("katere vrhove hoces dodat, ko napises zadnjega napis se OK:")
    while a != "OK":
        a = input()
        if a != "OK":
            lstdodaj.append(a)

    listastevil = np.array([])
    for i in range(len(lstdodaj)):  # loci listo z vrhovi in valovnim stevilom na ime vrha npr. VS  in pa valovno stevilo
        listastevil = np.append(listastevil, lstdodaj[i])
        intenziteta = np.append(intenziteta, 100*(y[find_nearest(x, float(lstdodaj[i]))]-doliney[find_nearest(dolinex, float(lstdodaj[i]))]))

    for i in range(len(listastevil)):
        vrhovix = np.append(vrhovix, x[find_nearest(x, int(listastevil[i]))])
        vrhoviy = np.append(vrhoviy, y[find_nearest(x, int(listastevil[i]))])
        if intenziteta[i] > VSP:
            listx.append(str(round(x[find_nearest(x, int(listastevil[i]))])) + "VS_P")
        elif intenziteta[i] > SP:
            listx.append(str(round(x[find_nearest(x, int(listastevil[i]))])) + "S_P")
        elif intenziteta[i] > MP:
            listx.append(str(round(x[find_nearest(x, int(listastevil[i]))])) + "M_P")
        elif intenziteta[i] > WP:
            listx.append(str(round(x[find_nearest(x, int(listastevil[i]))])) + "W_P")
        elif intenziteta[i] > VWP:
            listx.append(str(round(x[find_nearest(x, int(listastevil[i]))])) + "VW_P")
        else:
            listx.append(str(round(x[find_nearest(x, int(listastevil[i]))])) + "Sh")

    for i in range(len(vrhovix) - 1):
        plt.annotate(round(intenziteta[i]), (vrhovix[i + 1]-20, vrhoviy[i + 1]+0.01))  # numpy arraye appenda drgac k pa obicne don't ask
        plt.annotate(listx[i], (vrhovix[i + 1]-20, vrhoviy[i + 1] + 0.03))
    plt.scatter(vrhovix[1:], vrhoviy[1:], color='red')
    plt.plot(x, y)
    plt.grid()
    plt.title(ime_grafa)
    plt.xlabel('val_stevilo(1/cm)')
    plt.ylabel('normalizirana_intenziteta')
    plt.show()

    return x, y, vrhovix, vrhoviy, listx

def pisanje_mapa(x, y, vrhovix, vrhoviy, listx, intenziteta):
    new_list = [x, y, vrhovix, vrhoviy, listx, intenziteta]
    d  = dict(x=x, y=y, vrhovix=vrhovix, vrhoviy=vrhoviy, intenziteta=intenziteta)
    df = pd.DataFrame.from_dict(d, orient='index').transpose().fillna('')
    df.to_csv(mapaodp+ime+'.csv')


def zrisanje(ime, podatki, izbrisraileigha, kolk_jih_zgladi, VSP, SP, MP, WP, VWP, Sh, mera_za_shoulderje, mera_za_dvojne_shoulderje, glajenje, winer, ime_grafa):
    valstkompozit, intenzitetakompozit = odpiranje_podatkov(ime, st_podatkov)[0], odpiranje_podatkov(ime, st_podatkov)[
        1]
    spekter = raman(valstkompozit, intenzitetakompozit, podatki, izbrisraileigha, kolk_jih_zgladi, VSP, SP, MP, WP, VWP,
                    Sh, mera_za_shoulderje, mera_za_dvojne_shoulderje, winer)
    plt.plot(spekter.x, spekter.y)
    spekter.premikrayleigha()
    plt.title('Neobdelan')
    plt.xlabel('val_stevilo(1/cm)')
    plt.ylabel('intenziteta')
    plt.grid()
    plt.show()
    spekter.gama()
    spekter.pobris_raileigha()
    spekter.fitozad()
    spekter.normalizacija()
    plt.plot(spekter.x, spekter.y)
    if glajenje == 1:
        spekter.glajenje()
    spekter.wiener()
    listx, vrhovix, vrhoviy, intenziteta, dolinex, doliney = spekter.vrhovi_in_doline()
    for i in range(len(vrhovix) - 1):
        plt.annotate(round(intenziteta[i]), (vrhovix[i + 1], vrhoviy[i + 1]+0.1))  # numpy arraye appenda drgac k pa obicne don't ask
        plt.annotate(listx[i], (vrhovix[i+1], vrhoviy[i+1]+0.08))
    plt.scatter(vrhovix[1:], vrhoviy[1:]+0.1, color='red')
    plt.plot(spekter.x, spekter.y)
    plt.grid()
    plt.title('Polovicno in skoraj celotno obdelan')
    plt.xlabel('val_stevilo(1/cm)')
    plt.ylabel('normalizirana_intenziteta')
    plt.show()
    print(np.shape(vrhovix), np.shape(vrhoviy), np.shape(listx))
    vrhovix, vrhoviy, listx = zbrisanje(vrhovix, vrhoviy, listx)
    print(np.shape(vrhovix), np.shape(vrhoviy), np.shape(listx))
    spekter.x, spekter.y, vrhovix, vrhoviy, listx = dodajanje(spekter.x, spekter.y, vrhovix, vrhoviy, listx, intenziteta, dolinex, doliney, VWP, WP, MP, SP, VSP, ime_grafa)
    print(np.shape(vrhovix), np.shape(vrhoviy), np.shape(listx))
    pisanje_mapa(spekter.x, spekter.y, vrhovix, vrhoviy, listx, intenziteta)
    return spekter.x, spekter.y, listx, vrhovix, vrhoviy, intenziteta

# atributi ki jih lahko stimas za class
zvisanje = 1
VWP = zvisanje*0.05  # kolk odstopanja rab bit za very weak peak
WP = zvisanje*0.1  # weak peak itd
MP = zvisanje*0.15
SP = zvisanje*0.2
VSP = zvisanje*0.3
Sh = zvisanje*0.2
mera_za_shoulderje = 0.2  # kolk more bit nizij od peaka da je to shoulder

mera_za_dvojne_shoulderje = 0.02  # do kaksne razlike sta to ce je en shoulder potem oba shoulderja
izbrisraileigha = 66  # kolk podatkou zbrise (te je dokaj arbitrarno, sam na zacetku je bl k ne sam raileigh in je mal pointless)
kolk_jih_zgladi = 3  # kolk jih skupej zgladi, trenutno zarad RBF to ni ful potrebno
st_podatkov = 1024   # stevilo podatkou
glajenje = 1  # ce je 1 bo naredil glajenje, cene pa ne
winer = 7 # kriterij kolk sosedov gleda za wienerjev fit - tem vec tem bl fita, ampak tut popac spekter

ime = 'test'
mapaodp = 'E:\\Raman\\'
mapa_za_odpiranje = mapaodp + ime +'.txt'  # ime datotekbe z lokacijo
ime_grafa = 'Ramanski spekter na otoku'

x, y, listx, vrhovix, vrhoviy, intenziteta = zrisanje(mapa_za_odpiranje, st_podatkov, izbrisraileigha, kolk_jih_zgladi, VSP, SP, MP, WP, VWP, Sh, mera_za_shoulderje, mera_za_dvojne_shoulderje, glajenje, winer, ime_grafa)
