import numpy as np
import os
import time
from mpi4py import MPI

c1=time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a=[]
with open ('nbrlist/bonds.out', 'r') as f:
    for line in f:
        a.append(line.split())
                 
nats=int(a[3][0])

x = [[] for i in range(324)]

for i in a[8+1:8+nats+1]:
    x[int(i[0])-1].append(int(i[1]))
    x[int(i[1])-1].append(int(i[0]))
                                            
hbonds=np.array(x[108:])
#print(hbonds.shape)
#print(hbonds)

a=[]
with open ('nbrlist/nbr1s.out', 'r') as f:
    for line in f:
        a.append(line.split())

nats=int(a[3][0])

x = [[] for i in range(324)]

for i in a[8+1:8+nats+1]:
    x[int(i[0])-1].append(int(i[1]))
    x[int(i[1])-1].append(int(i[0]))
                                    
hnn1s=np.sort(np.array(x[108:]))[:,hbonds.shape[1]:]
print(hnn1s.shape)
#print(hnn1s)

a=[]
with open ('nbrlist/nbr2s.out','r') as f:
    for line in f:
        a.append(line.split())

nats=int(a[3][0])

x=[[] for i in range(324)]

for i in a[8+1:8+nats+1]:
    x[int(i[0])-1].append(int(i[1]))
    x[int(i[1])-1].append(int(i[0]))

hnn2s=np.sort(np.array(x[108:]))

nn2s=[]
for i in range(hnn2s.shape[0]):
    nn2s.append(np.setdiff1d(hnn2s[i],np.concatenate((hnn1s[i],hbonds[i]))))

hnn2s=np.array(nn2s)
print(hnn2s.shape)

a=[]
with open ('nbrlist/nbr3.out','r') as f:
    for line in f:
        a.append(line.split())
        
nats=int(a[3][0])

x=[[] for i in range(324)]

for i in a[8+1:8+nats+1]:
    x[int(i[0])-1].append(int(i[1]))
    x[int(i[1])-1].append(int(i[0]))
    
hnn3=np.sort(np.array(x[108:]))

nn3=[]
for i in range(hnn3.shape[0]):
    nn3.append(np.setdiff1d(hnn3[i],np.concatenate((hnn2s[i],hnn1s[i],hbonds[i]))))

hnn3=np.array(nn3)
hnn3=hnn3[:,12:]
nn3=[]
for i in range(216):
    temp=[]
    for j in hnn3[i]:
        if len(np.setdiff1d(hbonds[j-109],hbonds[i]))==4:
            temp.append(j)
    nn3.append(temp)
hnn3=np.array(nn3)
print(hnn3.shape)
#print(hnn3[67])

#exit()
NN_all=np.column_stack((hnn1s,hnn2s,hnn3))

a=[]
with open ('nbrlist/data.yh2-3', 'r') as f:
    for line in f:
        a.append(line.split())

hyd_pos=np.array(a[125:],dtype='float')[:,2:]

nn3_d=np.sqrt((float(a[5][1])/3)**2+(float(a[6][1])/3)**2+(float(a[7][1])/3)**2)/2
dim=np.array([float(a[5][1]),float(a[6][1]),float(a[7][1])])
#print(dim)
def pbc_dist(i,j,nx,ny,nz):
    global hyd_pos
    import numpy as np
    disp=hyd_pos[j]-hyd_pos[i]
    delta=2*nn3_d
    if disp[0]>delta:
        nx=nx-1
    if disp[0]<-delta:
        nx=nx+1
    if disp[1]>delta:
        ny=ny-1
    if disp[1]<-delta:                      
        ny=ny+1
    if disp[2]>delta:
        nz=nz-1      
    if disp[2]<-delta:
        nz=nz+1
    return(nx,ny,nz,hyd_pos[j]+dim*np.array([nx,ny,nz]))

start_site=311
nsim=100
temperature=1000
timesteps=10**3
def kmc(T):
    import numpy as np
    global start_site
    init=np.array([start_site-109])
    nx,ny,nz=0,0,0
    start_time=[nx,ny,nz]
    k=8.617*10**-5
    freq=np.array([1.193528,1.575398,2.062862])*10**13
    bar=np.array([0.862278,0.898660,0.934282])
    nbrs=[6,12,4]
    global NN_all
    rate_path=freq*np.exp(-bar/(k*(T+273)))
    rates_all=np.array([])
    rates_all=np.append(rates_all,np.repeat(rate_path,nbrs))
    rates_total=np.cumsum(rates_all)
    rates_prob=rates_total/rates_total[-1]
    timesteps=10**3
    time_real=np.array([0])
    posit=np.array([])
    posit=np.append(posit,np.array(hyd_pos[init]))
    n1,n2,n3=0,0,0
    for t in range(timesteps):
        r=np.random.rand(1,1)[0,0]
        select=np.searchsorted(rates_prob, [r])
        if select <=5:
            n1=n1+1
        elif 6<=select<=17:
            n2=n2+1
        elif 18<=select:
            n3=n3+1
        select_site=NN_all[init[t]][select]
        init=np.append(init,select_site-109)
        nx,ny,nz,pos_new=pbc_dist(init[t],init[t+1],nx,ny,nz)
        posit=np.vstack((posit,pos_new))
        time_real=np.append(time_real,-np.log(np.random.rand(1,1)[0,0])/rates_total[-1])
    end_time=[nx,ny,nz]
    return time_real, posit, timesteps, np.array(start_time), np.array(end_time), np.array([n1,n2,n3])

#c1=time.time()
#output=[kmc(temperature) for i in range(10)]
#print(time.time()-c1)

#cent=np.tile(np.array([np.empty(100,dtype='float'),np.empty((100,3),dtype='float'),[],[],[],[]],dtype='object'),10)
#recvbuf=np.array([np.empty(100,dtype='float'),np.empty((100,3),dtype='float'),[],[],[],[]],dtype='object')
cent=np.ones(10)
recvbuf=np.empty(1)

comm.Scatter(cent,recvbuf,root=0)
print('scattered')
#output=[]
out0=np.empty((1,timesteps+1))
out1=np.empty((1,timesteps+1,3))
out3=np.empty((1,3),dtype='int')
out4=np.empty((1,3),dtype='int')
out5=np.empty((1,3),dtype='int')
for i in range(int(nsim/size)):
    output=kmc(temperature)
    print('calculated')
    out0=np.vstack((out0,np.array(output[0]).reshape(1,timesteps+1)))
    out1=np.vstack((out1,np.array(output[1]).reshape(1,timesteps+1,3)))
    #out0=np.vstack((out0,output[0]))
    out3=np.vstack((out3,np.array(output[3]).reshape(1,3)))
    out4=np.vstack((out4,np.array(output[4]).reshape(1,3)))
    out5=np.vstack((out5,np.array(output[5]).reshape(1,3)))
out0=out0[1:]
out1=out1[1:]
out3=out3[1:]
out4=out4[1:]
out5=out5[1:]
#print("out0",out3.shape)
#print(output)
#output=np.array(output,dtype='object')
print('calculation complete')

o0,o1,o3,o4,o5=None,None,None,None,None
if rank==0:
    #tbuf=np.tile(np.array([np.empty(100,dtype='float'),np.empty((100,3),dtype='float'),[],[],[],[]],dtype='object'),10)
    o0=np.empty((nsim,timesteps+1))
    #print(o0.shape)
    o1=np.empty((nsim,timesteps+1,3))
    #print(output[2])
    #o2=[]*10
    o3=np.empty((nsim,3),dtype='int')
    o4=np.empty((nsim,3),dtype='int')
    o5=np.empty((nsim,3),dtype='int')

comm.Gather(out0,o0,root=0)
comm.Gather(out1,o1,root=0)
#comm.Gather(output[2],o2,root=0)
comm.Gather(out3,o3,root=0)
comm.Gather(out4,o4,root=0)
comm.Gather(out5,o5,root=0)
#print(o3.shape)
if rank==0:
    #o0=np.reshape(o0,(10,101))
    print('gathered')
    c2=time.time()
    print('time for calculation=',c2-c1,'s')
    #print(o0[0])
    #o1=np.reshape(o1,(101,3,10))
    #print(o5)
    c1=time.time()
    print('writing data to files')
    path='T='+str(temperature)
    if os.path.exists(path)==False:
        os.mkdir(path)
    for i in range(nsim):
        np.savetxt(path+'/'+str(i)+'.txt',np.column_stack((o0[i],o1[i])))
        np.savetxt(path+'/'+str(i)+'extra.txt',np.concatenate((o3[i],o4[i],o5[i])).reshape(1,9),fmt='%i')

    c2=time.time()
#print('time for calculation =',c2-c1,'s')
    print('time for writing=',c2-c1,'s')

#print(o0[0])

#print(output)
#c1=time.time()
#res = direct_view.map(kmc, nsim*[temperature])
#kmc_output=res.result()
#c2=time.time()
#print(c2-c1)
#print(len(kmc_output))
#print(kmc_output[0][0].shape)

#path='T='+str(temperature)
#os.makedirs(path,exist_ok=False)
#for i in range(nsim):
#    np.savetxt(path+'/'+str(i)+'.txt',np.column_stack((o0[i],o1[i])))
