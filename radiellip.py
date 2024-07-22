import numpy as np
import healpy as hp
import copy
import math
import os
os.chdir('/project/ls-mohr/users/xu.han/voids')

#Basic parameters
nside = 1024
npix = hp.nside2npix(nside)
A = 129600/(np.pi*npix)

def projection(ind, centroids_pos):
    voids_pos = hp.pix2ang(nside, voids[ind], lonlat = False)
    v = np.asarray(hp.rotator.dir2vec(voids_pos[0], phi=voids_pos[1], lonlat=False)).T
    v_cen = hp.rotator.dir2vec(centroids_pos[ind][0], phi=centroids_pos[ind][1], lonlat=False)
    #projected vectors
    v_p = np.empty([len(v),3])
    for i in range(len(v)):
        v_p[i] = np.subtract(v[i], np.multiply(np.sum(np.multiply(v[i],v_cen)), v_cen))
    length = np.sqrt(v_p[0][0]**2+v_p[0][1]**2+v_p[0][2]**2)
    #basis vectors
    v_b1 = np.array([v_p[0][0]/length,v_p[0][1]/length,v_p[0][2]/length])
    v_b2 = np.cross(v_cen,v_b1)
    v_new = np.empty([len(v_p),2])
    for i in range(len(v_p)):
        v_new[i][0] = np.dot(v_p[i],v_b1)
        v_new[i][1] = np.dot(v_p[i],v_b2)
    return v_new

def radi_elli(voids):
    num = np.array([], dtype=int)
    for i in voids:
        num = np.append(num, len(i))
    radi = np.sqrt(num*A/np.pi)
    #Centroid coordinates in radian
    centroids_pos = np.empty([len(voids),2])
    centroids_pix = np.empty(len(voids), dtype=int)
    for i in range(len(voids)):
        voids_pos = hp.pix2ang(nside, voids[i], lonlat = False)
        v = hp.rotator.dir2vec(voids_pos[0], phi=voids_pos[1], lonlat=False)
        cen_vec = np.asarray([np.mean(v[0]), np.mean(v[1]), np.mean(v[2])])
        #cen_pos = hp.rotator.vec2dir(cen_vec, lonlat=False)
        #cen_pix = hp.ang2pix(nside, cen_pos[0], cen_pos[1], lonlat=False)
        centroids_pos[i] = hp.rotator.vec2dir(cen_vec, lonlat=False)
        centroids_pix[i] = hp.ang2pix(nside, centroids_pos[i][0], centroids_pos[i][1], lonlat=False)
        
    #projection of voids and ellipticity
    ellip = []
    for i in range(len(voids)):
        if len(voids[i])!=1:
            v_2d = projection(i, centroids_pos)
            cen_0 = np.mean(v_2d[:,0])
            cen_1 = np.mean(v_2d[:,1])
            S00 = np.mean(np.multiply(v_2d[:,0]-cen_0, v_2d[:,0]-cen_0))
            S01 = np.mean(np.multiply(v_2d[:,0]-cen_0, v_2d[:,1]-cen_1))
            S10 = np.mean(np.multiply(v_2d[:,1]-cen_1, v_2d[:,0]-cen_0))
            S11 = np.mean(np.multiply(v_2d[:,1]-cen_1, v_2d[:,1]-cen_1))
            l1 = 0.5*(S00+S11+np.sqrt((S00-S11)**2+4*S01*S10))
            l2 = 0.5*(S00+S11-np.sqrt((S00-S11)**2+4*S01*S10))
            ellip.append(1-np.sqrt((l2/l1)))
        else:
            ellip.append(0)

        
    return radi, ellip

rz = [0,1,2,3,4,5]
rs = [35]
sl = [10,5,2.5,1]

for n0 in rz:
    for n1 in rs:
        for n2 in sl:
            voids = np.load('r00%s/voids_00%s_z%s_1024_%s.npy'%(n0,n0,n1,n2), allow_pickle=True)
            radi, ellip = radi_elli(voids)
            np.save('r00%s/radi_00%s_z%s_1024_%s.npy'%(n0,n0,n1,n2), radi)
            np.save('r00%s/ellip_00%s_z%s_1024_%s.npy'%(n0,n0,n1,n2), ellip)
            print('realization=',n0,'zs=',n1,'sl=',n2,'is finished.', end='\r')