import numpy as np
import healpy as hp
import os

os.chdir('/project/ls-mohr/users/xu.han/voids')

rz = [3,4,5]
sl = [10,5,2.5,1]

#function for profile, ind means the index of voids, sl means step length (in arcmin)
def prof_all(ind, sl):
    v = hp.rotator.dir2vec(centroids_pos[ind][0], phi=centroids_pos[ind][1], lonlat=False)
    s=0
    void = voids[ind]
    profile = []
    pix = np.asarray([centroids_pix[ind]])
    pix_t = np.asarray([centroids_pix[ind]])
    #define how many steps from centroid
    while s<50:
        profile.append(pix)
        pix0 = hp.query_disc(nside, v, sl*s*np.pi/180)
        pix1 = hp.query_disc(nside, v, sl*(s+1)*np.pi/180)
        pix01 = np.setdiff1d(pix1, pix0)
        a = np.append(pix01, void)
        u, c = np.unique(a, return_counts=True)
        pix_t = u[c > 1]
        pix = pix01
        s += 1
    #Remove center pixel
    index = np.argwhere(profile[1]==profile[0][0])
    profile[1] = np.delete(profile[1], index)
    
    return profile

def prof_rel(ind, step, r_ind):
    v = hp.rotator.dir2vec(centroids_pos[ind][0], phi=centroids_pos[ind][1], lonlat=False)
    s=0
    void = voids[ind]
    profile = []
    pix = np.asarray([centroids_pix[ind]])
    sl = 4*radius[r_ind]/step
    while s != step+1:
        profile.append(pix)
        pix0 = hp.query_disc(nside, v, sl*s*np.pi/180)
        pix1 = hp.query_disc(nside, v, sl*(s+1)*np.pi/180)
        pix01 = np.setdiff1d(pix1, pix0)
        pix = pix01
        s += 1
    #Remove center pixel
    index = np.argwhere(profile[1]==profile[0][0])
    profile[1] = np.delete(profile[1], index)
    
    return profile

it = 0
for n0 in rz:
    for n1 in sl:
        kappa_masked_smooth_load = hp.read_map('r00%s/kappa_masked_smooth_00%s_zs35_1024_%s.fits'%(n0,n0,n1))
        kappa_masked_smooth = hp.ma(kappa_masked_smooth_load)
        nside = hp.get_nside(kappa_masked_smooth)
        npix = hp.nside2npix(nside)
        voids = np.load('r00%s/voids_00%s_merged0.5_z35_1024_%s.npy'%(n0,n0,n1), allow_pickle=True)
        #Centroid coordinates in radian
        centroids_pos = np.empty([len(voids),2])
        centroids_pix = np.empty(len(voids), dtype=int)
        for i in range(len(voids)):
            voids_pos = hp.pix2ang(nside, voids[i], lonlat = False)
            v = hp.rotator.dir2vec(voids_pos[0], phi=voids_pos[1], lonlat=False)
            cen_vec = np.asarray([np.mean(v[0]), np.mean(v[1]), np.mean(v[2])])
            centroids_pos[i] = hp.rotator.vec2dir(cen_vec, lonlat=False)
            centroids_pix[i] = hp.ang2pix(nside, centroids_pos[i][0], centroids_pos[i][1], lonlat=False)
            
        #remove the voids with a centroid outside 
        ignor = []
        for i in range(len(centroids_pos)):
            if centroids_pix[i] not in voids[i]:
                ignor.append(i)
                
        Pro_all = []
        for i in range(len(centroids_pos)):
            if i not in ignor:
                prof_void = prof_all(i, 0.05)
                Pro_all.append(prof_void)
                
        length_all = []
        for i in Pro_all:
            length_all.append(len(i))
        number_all = np.max(length_all)
        kp_all = np.zeros([np.max(length_all),2])
        for i in range(len(Pro_all)):
            for j in range(length_all[i]):
                if len(Pro_all[i][j]) != 0:
                    kp_all[j][0] = kp_all[j][0]+np.sum(kappa_masked_smooth[Pro_all[i][j]])
                    kp_all[j][1] += len(Pro_all[i][j])
                    
        np.save('r00%s/prof_cen_0.05_00%s_merged0.5_z35_1024_%s.npy'%(n0,n0,n1), kp_all)
        
        num_pix = []
        radius = []
        A = 129600/(np.pi*npix)
        for i in voids:
            num_pix.append(len(i))
            radius.append(np.sqrt(len(i)*A/np.pi))
            
        r_s = []
        for i in range(len(radius)):
            r_s.append(radius[i]**2)
        weight = r_s/np.mean(r_s)
            
        step = 40
        Pro_rel = []
        rad_num = 0
        for i in range(len(centroids_pos)):
            if i not in ignor:
                prof_void = prof_rel(i, step, rad_num)
                Pro_rel.append(prof_void)
                rad_num += 1
                
        #length_rel = []
        #for i in Pro_rel:
            #length_rel.append(len(i))
    
        #kp_rel = np.zeros([step,2])
        #for i in range(len(Pro_rel)):
            #for j in range(step):
                #if len(Pro_rel[i][j])!=0:
                    #kp_rel[j][0] = kp_rel[j][0]+np.sum(kappa_masked_smooth[Pro_rel[i][j]])
                    #kp_rel[j][1]+=len(Pro_rel[i][j])
        # with weights 
        length_rel = []
        for i in Pro_rel:
            length_rel.append(len(i))
            
        kp_rel = np.zeros([step,2])
        for i in range(len(Pro_rel)):
            for j in range(step):
                if len(Pro_rel[i][j])!=0:
                    kp_rel[j][0] = kp_rel[j][0]+(np.mean(kappa_masked_smooth[Pro_rel[i][j]])*weight[i])
                    kp_rel[j][1] += 1
            #if i%100==0:
                #print(i, end='\r')            
        np.save('r00%s/prof_cen_re_4_40_00%s_merged0.5_z35_1024_%s_new1.npy'%(n0,n0,n1), kp_rel)
        print(it, end='\r')
        it += 1