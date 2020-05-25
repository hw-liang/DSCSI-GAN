import torch 
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
import numpy as np
import pytorch_colors as colors

def rgb2lab(imageTensor):
    # (b,c,h,w) -> (b,h,w,c) -> (b,c,h,w)
    imageTensor = imageTensor.permute(0,2,3,1)
    tmp = colors.rgb_to_lab(imageTensor).float()
    labImg = tmp.permute(0,3,1,2)
    return labImg

def gaussianWindow(windowSize, sigma):
    gauss1d = torch.Tensor([exp(-(x - windowSize//2)**2/float(2*sigma**2)) for x in range(windowSize)]).unsqueeze(1)
    gauss2d = gauss1d.mm(gauss1d.t()).float().unsqueeze(0).unsqueeze(0)/(2*np.pi*sigma)
    gauss2d /= gauss2d.sum() # normalized gaussian
    window = Variable(gauss2d.contiguous().cuda(), requires_grad = False)
    return  window

def hue_chromaComponents(inputA,inputB):
    Xh = torch.atan(inputB/inputA)
    Xc = torch.sqrt(inputA**2+inputB**2)
    return Xh,Xc

def circularMeanVariance(inputXh,windowSize,window):
    sin = torch.sin(inputXh).unsqueeze(1)
    cos = torch.cos(inputXh).unsqueeze(1)
    weightedSin = F.conv2d(sin, window, padding = windowSize//2)
    weightedCos = F.conv2d(cos, window, padding = windowSize//2)
    hMean = torch.atan(weightedSin/weightedCos)
    midCos = F.conv2d(cos**2,window**2,padding = windowSize//2)
    midSin = F.conv2d(sin**2,window**2,padding = windowSize//2)
    hVar = 1 - torch.sqrt(midCos + midSin)/windowSize**2
    return hMean,hVar

def aFunc(Xc1,Xc2,c0,l):
    return 0.5+0.5*torch.tanh((torch.min(Xc1,Xc2)-c0)/(l*c0))

def fFunc(diff,h0,tao):
    return 0.5+0.5*torch.tanh((diff-h0)/(tao*h0))

def hl_hcFunc(hMean1,hMean2,hVar1,hVar2,Xc1,Xc2,KH):
    diff = np.pi-torch.abs(np.pi-torch.abs(hMean1-hMean2))
    # a = aFunc(Xc1,Xc2,10,0.25)
    hl = (1-fFunc(diff,0.2*np.pi,0.35))  # discard "multiply a"
    hc = ((2*hVar1*hVar2+KH)/(hVar1**2+hVar2**2+KH)) # discard "multiply a"
    return hl,hc

def meanStandardDeviation(inputXc,windowSize,window):
    inputXc = inputXc.unsqueeze(1)
    cMean = F.conv2d(inputXc,window,padding = windowSize//2)
    cStd = F.conv2d((inputXc-cMean)**2,window,padding = windowSize//2)**0.5
    return cMean,cStd

def cl_ccFunc(cMean1,cMean2,cStd1,cStd2,KC1,KC2):
    cl = 1/(KC1*(cMean1-cMean2)**2+1)
    cc = (2*cStd1*cStd2+KC2)/(cStd1**2+cStd2**2+KC2)
    return cl,cc

def lVarianceCovariance(inputL1,inputL2,windowSize,window):
    inputL1 = inputL1.unsqueeze(1)
    inputL2 = inputL2.unsqueeze(1)
    lmean1 = F.conv2d(inputL1,window,padding = windowSize//2)
    lmean2 = F.conv2d(inputL2,window,padding = windowSize//2)
    lvar1 = F.conv2d((inputL1-lmean1)**2,window,padding = windowSize//2)**0.5
    lvar2 = F.conv2d((inputL2-lmean2)**2,window,padding = windowSize//2)**0.5
    lCovar = F.conv2d((inputL1-lmean1)*(inputL2-lmean2),window,padding = windowSize//2)
    return lvar1,lvar2,lCovar

def lc_lsFunc(lvar1,lvar2,lCovar,KL1,KL2):
    lc = (2*lvar1*lvar2+KL1)/(lvar1**2+lvar2**2+KL1)
    ls = (torch.abs(lCovar)+KL2)/(lvar1*lvar2+KL2)
    return lc,ls

def avgPooling(input,p):
    output = 1-(torch.sum((1-input)**p,dim = (2,3))/(input.size()[2]*input.size()[3]))**(1/p)
    return output

def DSCSI(Hl,Hc,Cl,Cc,Lc,Ls,LAMBDA):
    SC = Hl*Hc*Cl*Cc
    SA = Lc*Ls
    DSCIS_score = SA*SC**LAMBDA
    return DSCIS_score

class COLOR_DSCSI(torch.nn.Module):
    def __init__(self,windowSize,sigma):
        super(COLOR_DSCSI, self).__init__()
        self.windowSize = windowSize
        self.sigma = sigma
        self.window = gaussianWindow(self.windowSize,self.sigma)
    
    def forward(self,img1,img2):
        labImg1 = rgb2lab(img1)
        labImg2 = rgb2lab(img2)

        Xh1,Xc1 = hue_chromaComponents(labImg1[:,1],labImg1[:,2])
        Xh2,Xc2 = hue_chromaComponents(labImg2[:,1],labImg2[:,2])

        hMean1,hVar1 = circularMeanVariance(Xh1,self.windowSize,self.window)
        hMean2,hVar2 = circularMeanVariance(Xh2,self.windowSize,self.window)
        hl,hc = hl_hcFunc(hMean1,hMean2,hVar1,hVar2,Xc1,Xc2,0.0008)

        cMean1,cstd1 = meanStandardDeviation(Xc1,self.windowSize,self.window)
        cMean2,cstd2 = meanStandardDeviation(Xc2,self.windowSize,self.window)
        cl,cc = cl_ccFunc(cMean1,cMean2,cstd1,cstd2,0.0008,16)

        lVar1,lvar2,lCovar = lVarianceCovariance(labImg1[:,0],labImg2[:,0],self.windowSize,self.window)
        lc,ls = lc_lsFunc(lVar1,lvar2,lCovar,0.8,0.8)

        Hl,Hc =  avgPooling(hl,1),avgPooling(hc,1)
        Cl,Cc =  avgPooling(cl,1),avgPooling(cc,1)
        Lc,Ls =  avgPooling(lc,1),avgPooling(ls,1)

        DSCIS_score = DSCSI(Hl,Hc,Cl,Cc,Lc,Ls,0.8)

        return DSCIS_score.sum()/ img1.size(0)

