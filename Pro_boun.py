import numpy as np
import healpy as hp
import math
import os
from collections import Counter
from sklearn.metrics.pairwise import haversine_distances
os.chdir('/project/ls-mohr/users/xu.han/voids')

rz = [0]
sl = [10,5,2.5,1]

#calculate distance from pixel to boundary
def dis_in_out(ind):
    void = voids[ind]
    bound = voids_bound[ind]
    lonlat_b = np.asarray(hp.pix2ang(nside, bound, lonlat=True)).T*np.pi/180
    bound_pos = np.zeros([len(lonlat_b), 2])
    bound_pos[:,1] = lonlat_b[:,0]
    bound_pos[:,0] = lonlat_b[:,1]
    
    #pix = void
    pix_all = np.zeros(npix,dtype=bool)
    for i in bound:
        pos = np.asarray(hp.pix2ang(nside, i, lonlat=True))
        v = hp.rotator.dir2vec(pos[0], phi=pos[1], lonlat=True)
        query = np.asarray(hp.query_disc(nside, v, 2*radius[ind]))       
        pix_all[query] = True
        
    pix = np.asarray(np.nonzero(pix_all))[0]
    
    lonlat_pix = np.asarray(hp.pix2ang(nside, pix, lonlat=True)).T*np.pi/180
    pix_pos = np.zeros([len(lonlat_pix), 2])
    pix_pos[:,1] = lonlat_pix[:,0]
    pix_pos[:,0] = lonlat_pix[:,1]

    
    dis = haversine_distances(pix_pos, bound_pos)
    dis_bound = np.empty([len(pix),2])  #First column is pixel number, second is distance (in radian)
    for i in range(len(pix)):
        indice = np.argmin(dis[i], axis=None)
        dis_bound[i][0] = pix[i]
        if pix[i] in void:
            dis_bound[i][1] = -1 * dis[i][indice]
        else:
            dis_bound[i][1] = dis[i][indice]
        
    return dis_bound

it = 0
for n0 in rz:
    for n1 in sl:
        kappa_masked_smooth_load = hp.read_map('r00%s/kappa_masked_smooth_00%s_zs35_1024_%s.fits'%(n0,n0,n1))
        kappa_masked_smooth = hp.ma(kappa_masked_smooth_load)
        nside = hp.get_nside(kappa_masked_smooth)
        npix = hp.nside2npix(nside)
        voids = np.load('r00%s/voids_00%s_merged0.5_z35_1024_%s.npy'%(n0,n0,n1), allow_pickle=True)
        voids_bound = np.load('r00%s/voids_bound_00%s_merged0.5_z35_1024_%s.npy'%(n0,n0,n1), allow_pickle=True)
        
        num_pix = []
        radius = []
        A = 129600/(np.pi*npix)
        for i in voids:
            num_pix.append(len(i))
            radius.append(np.sqrt(len(i)*A/np.pi)*np.pi/180)
            
        profile_b = []
        for i in range(len(voids)):
            dis_b = dis_in_out(i)
            profile_b.append(dis_b)
            
        k_pos = []
        k_neg = []
        for j in range(len(profile_b)):
            prof_pos = profile_b[j][np.where(profile_b[j][:,1] >= 0)[0]]
            prof_neg = profile_b[j][np.where(profile_b[j][:,1] < 0)[0]]
            prof_pos[:,1] = prof_pos[:,1]//(0.05*np.pi/180)
            prof_neg[:,1] = prof_neg[:,1]//(0.05*np.pi/180) + 1
            prof_pos = prof_pos.astype(int)
            prof_neg = prof_neg.astype(int)
            kappa_pos = np.zeros(np.max(prof_pos[:,1])+1) #first term is sum of kappa, second is number of values
            kappa_neg = np.zeros(-np.min(prof_neg[:,1]))
            for i in range(np.max(prof_pos[:,1])+1):
                kappa_pos[i] = np.mean(kappa_masked_smooth[prof_pos[np.where(prof_pos[:,1]==i)[0]][:,0]])
            for i in range(-np.min(prof_neg[:,1])):
                kappa_neg[i] = np.mean(kappa_masked_smooth[prof_neg[np.where(prof_neg[:,1]==-(i+1))[0]][:,0]])
            k_pos.append(kappa_pos)
            k_neg.append(kappa_neg)
            
        length_neg = []
        for i in k_neg:
            length_neg.append(len(i))
        k_neg_t = np.zeros([np.max(length_neg), 2])
        k_neg_mean = np.zeros(np.max(length_neg))
        for i in k_neg:
            for j in range(len(i)):
                if math.isnan(i[j]):
                    continue
                else:
                    k_neg_t[j][0] = k_neg_t[j][0]+i[j]
                    k_neg_t[j][1] += 1
        
        length_pos = []
        for i in k_pos:
            length_pos.append(len(i))
        k_pos_t = np.zeros([np.max(length_pos), 2])
        k_pos_mean = np.zeros(np.max(length_pos))
        for i in k_pos:
            for j in range(len(i)):
                if math.isnan(i[j]):
                    continue
                else:
                    k_pos_t[j][0] = k_pos_t[j][0]+i[j]
                    k_pos_t[j][1] += 1
        it += 1            
        np.save('r00%s/Profile_bound_p_0.05_00%s_merged0.5_z35_1024_%s.npy'%(n0,n0,n1), k_pos_t, allow_pickle=True)
        np.save('r00%s/Profile_bound_n_0.05_00%s_merged0.5_z35_1024_%s.npy'%(n0,n0,n1), k_neg_t, allow_pickle=True)
        print(it, end='\r')