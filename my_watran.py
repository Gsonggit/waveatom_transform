# !/usr/bin/env python
# 
#   1-D forward and inverse wave atom transform code modified by Gaosong from origion fwa1.m and iwa.m
# 
#  parameter:
#     data —— numpy.ndarray , the origion 1-D signal
#     pat  —— str  ,  'p' or 'q',specifies the type of frequency partition satsfies parabolic scaling
#     tp   —— str  ,  'ortho(orthobasis)' or 'comp(complex)',specifies the type of transform
#
#     c    —— list   , c[i][j][k] is cofficient at scale i ,frequency index j,spatial index k
#
# Date:2019/1/15 
#
# Copyright (c) by Gaosong

import numpy as np
import math

def wat(data,pat='p',tp='ortho'):
    if len(data.shape) > 1 or data.shape[0] <= 32 :
        raise TypeError('input data invalid ,please check!')

    #check datalength is a power of 2
    mdata=check_length(data)
    N=len(mdata)
    if tp == 'ortho' or tp == 'orthobasis':
        lst=freq_partition(N/2,pat)
        f=np.fft.fft(mdata)/np.sqrt(N)
        c=[]
        for i in range(len(lst)):
            nw=len(lst[i])
            c.append([])
            for j in range(nw):
                if lst[i][j] == 0:
                    c[i].append([])
                else :
                    B=2**i
                    D=2*B
                    Ict=j*B
                    if j % 2 == 0 :
                        Ifm=Ict-2/3*B
                        Ito=Ict+4/3*B
                    else:
                        Ifm=Ict-1/3*B
                        Ito=Ict+5/3*B
                    res=np.zeros(D,dtype='c8')
                    for k in range(2):
                        if k == 0 :
                            Idx=np.asarray([x for x in range(math.ceil(Ifm),math.floor(Ito)+1)])
                            Icf=kf_rt(Idx/B*np.pi,j)
                        else:
                            Idx=np.asarray([y for y in range(math.ceil(-Ito),math.floor(-Ifm)+1)])
                            Icf=kf_lf(Idx/B*np.pi,j)
                        res[Idx%D]=res[Idx%D]+np.conj(Icf)*f[Idx%N]
                    c[i].append(list((np.fft.ifft(res)*np.sqrt(res.shape[0])).real))
        return c
    elif tp == 'comp' or tp == 'complex' :
        lst=freq_partition(N/2,pat)
        f=np.fft.fft(mdata)/np.sqrt(N)
        c1=[]
        c2=[]
        for i in range(len(lst)):
            nw=len(lst[i])
            c1.append([])
            c2.append([])
            for j in range(nw):
                if lst[i][j] == 0:
                    c1[i].append([])
                    c2[i].append([])
                else :
                    B=2**i
                    D=2*B
                    Ict=j*B
                    if j % 2 == 0 :
                        Ifm=Ict-2/3*B
                        Ito=Ict+4/3*B
                    else:
                        Ifm=Ict-1/3*B
                        Ito=Ict+5/3*B   
                    res=np.zeros(D)  
                    Idx=np.asarray([x for x in range(math.ceil(Ifm),math.floor(Ito)+1)])
                    Icf=kf_rt(Idx/B*np.pi,j)  
                    res[Idx%D]=res[Idx%D]+np.conj(Icf)*f[Idx%N] 
                    c1[i].append(list((np.fft.ifft(res)*np.sqrt(res.shape[0]*res.shape[1])).real))

                    res=np.zeros((D,1))
                    Idx=np.asarray([y for y in range(math.ceil(-Ito),math.floor(-Ifm)+1)])
                    Icf=kf_lf(Idx/B*np.pi,j)
                    res[Idx%D]=res[Idx%D]+np.conj(Icf)*f[Idx%N]
                    c2[i].append(list((np.fft.ifft(res)*np.sqrt(res.shape[0]*res.shape[1])).real)) 
        c=[c1,c2]                                             
        return c

def iwa(c,pat='p',tp='ortho'):
    if tp == 'ortho' or tp == 'orthobasis':
        T=0
        for i in range(len(c)):
            nw=len(c[i])
            for j in range(nw):
                T=T+len(c[i][j])
        N=T
        lst=freq_partition(N/2,pat)
        f=np.zeros(N,dtype='c8')
        for i in range(len(lst)):
            nw=len(lst[i])
            for j in range(nw):
                if c[i][j] != []:
                    B=2**i
                    D=2*B
                    Ict=j*B
                    if j % 2 == 0 :
                        Ifm=Ict-2/3*B
                        Ito=Ict+4/3*B
                    else:
                        Ifm=Ict-1/3*B
                        Ito=Ict+5/3*B  
                    res=np.fft.fft(c[i][j])/np.sqrt(len(c[i][j]))
                    for k in range(2):
                        if k == 0 :
                            Idx=np.asarray([x for x in range(math.ceil(Ifm),math.floor(Ito)+1)])
                            Icf=kf_rt(Idx/B*np.pi,j)
                        else:
                            Idx=np.asarray([y for y in range(math.ceil(-Ito),math.floor(-Ifm)+1)])
                            Icf=kf_lf(Idx/B*np.pi,j)                
                        f[Idx%N]=f[Idx%N]+Icf*res[Idx%D]
        x=np.fft.ifft(f)*np.sqrt(len(f))
        return x.real

    elif tp == 'comp' or tp == 'complex' :
        c1=c[0]
        c2=c[1]
        T=0
        for i in range(len(c1)):
            nw=len(c1[i])
            for j in range(nw):
                T=T+len(c1[i][j])
        N=T
        lst=freq_partition(N/2,pat)
        f=np.zeros(N)
        for i in range(len(lst)):
            nw=len(lst[i])
            for j in range(nw):
                if c1[i][j] != []:
                    B=2**i
                    D=2*B
                    Ict=j*B
                    if j % 2 == 0 :
                        Ifm=Ict-2/3*B
                        Ito=Ict+4/3*B
                    else:
                        Ifm=Ict-1/3*B
                        Ito=Ict+5/3*B 
                    Idx=np.asarray([x for x in range(math.ceil(Ifm),math.floor(Ito)+1)])
                    Icf=kf_rt(Idx/B*np.pi,j)
                    res=np.fft.fft(c1[i][j])/np.sqrt(len(c1[i][j]))
                    f[Idx%N]=f[Idx%N]+Icf*res[Idx%D]
                    Idx=np.asarray([y for y in range(math.ceil(-Ito),math.floor(-Ifm)+1)])
                    Icf=kf_lf(Idx/B*np.pi,j)     
                    res=np.fft.fft(c2[i][j])/np.sqrt(len(c2[i][j]))
                    f[Idx%N]=f[Idx%N]+Icf*res[Idx%D]  
        x=np.fft.ifft(f)*np.sqrt(len(f))    
        return x.real


def check_length(data):
    if len(data) & (len(data)-1) == 0:
        return data
    else:
        nlen=2**((len(bin(len(data)))-2))
        at=np.zeros(nlen-len(data))
        data=np.hstack((data,at))
        return data


def freq_partition(length,pat):
    # return a frequency partition list
    if length < 16 :
        raise ValueError('length should be larger than 16 !! ')
    if pat == 'p':
        lst=[]
        lst.append([1,1])
        lst.append([0,1])
        lst.append([0,1,1,1])
        cnt=16
        idx=3
        rad=4
        while cnt < length :
            old=cnt
            lst[idx-1].extend([1,1])
            cnt+=2*rad
            idx+=1
            rad=2*rad
            trg=min(4*old,length)
            lst.append([0]*int(cnt/rad)+[1]*int((trg-cnt)/rad))
            cnt=trg
        return lst        
    elif pat == 'q' :
        lst=[]
        lst.append([1,1])
        lst.append([0,1,1,1])
        cnt=8
        idx=2
        rad=2
        while cnt < length :
            old=cnt
            lst[idx-1].extend([1,1])
            cnt+=2*rad
            idx+=1
            rad=2*rad
            trg=min(4*old,length)
            lst.append([0]*int(cnt/rad)+[1]*int((trg-cnt)/rad))
            cnt=trg
        return lst
    else :
        raise TypeError("please select from 'p' and 'q' !!")            

def kf_lf(w,n):
    #left bump
    r=np.zeros(len(w))
    an=np.pi/2*(n+1/2)
    en=(-1)**n
    en1=(-1)**(n+1)
    r=np.exp(-1j*w/2)*(np.exp(-1j*an)*g_func(en1*(w+np.pi*(n+1/2))))
    return r

def kf_rt(w,n):
    #right bump
    r=np.zeros(len(w))
    an=np.pi/2*(n+1/2)
    en=(-1)**n
    en1=(-1)**(n+1)
    r=np.exp(-1j*w/2)*(np.exp(1j*an)*g_func(en*(w-np.pi*(n+1/2))))
    return r

def g_func(w):
    r=np.zeros(len(w))
    gd=(w < 5*np.pi/6) & (w > -7*np.pi/6)
    r[gd]=abs(sf(w[gd]-3*np.pi/2))
    return r

def sf(w):
    r=np.zeros(len(w))
    aw=abs(w)
    r[aw <= 2*np.pi/3] = 0
    gd=(aw >= 2*np.pi/3) & (aw <= 4*np.pi/3)
    r[gd]=1/np.sqrt(2) * hf(w[gd]/2+np.pi)
    gd=(aw >= 4*np.pi/3) & (aw <= 8*np.pi/3)
    r[gd]=1/np.sqrt(2) * hf(w[gd]/4)
    r[aw > 8*np.pi/2] =0
    return r

def hf(w):
    w= (w+np.pi) % (2*np.pi) - np.pi
    r=np.zeros(len(w))
    w=abs(w)
    r=np.sqrt(2)*np.cos(np.pi/2*(1-np.cos(np.pi*(3*w/np.pi-1)))/2)
    r[w <= np.pi/3]=np.sqrt(2)
    r[w >= 2*np.pi/3]=0
    return r

# if __name__=='__main__':
#     from scipy import signal
#     t=np.linspace(0,10,5001)
#     w=signal.chirp(t,f0=12.5,f1=2.5,t1=10,method='linear')
#     rew=wat(w[0:4096],'p','ortho')
#     print(len(rew))
#     print(rew[0])
#     print(rew[1])
#     res=iwa(rew,'p','ortho')
#     print(res)