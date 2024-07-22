import numpy as np
import healpy as hp
import copy
from collections import Counter
import os
#os.chdir('/project/astro/simulations/CosmoGridConvergence/fiducial_baryonified')

def find_extrema(kappa_map,minima=False,lonlat=False, trim_edge_scale = 0, mask_dominated = False, return_mask_boundary = False):
    """find extrema in a smoothed masked healpix map
       default is to find peaks, finds minima with minima=True
    
       Parameters
       ----------
       kappa_masked_smooth: MaskedArray (healpy object)
           smoothed masked healpix map for which extrema are to be identified
       minima: bool
           if False, find peaks. if True, find minima
       
       Returns
       -------
       extrema_pos: np.ndarray
           extrema positions on sphere, theta and phi, in radians
       extrema_amp: np.ndarray
           extrema amplitudes in kappa
       
    """
    #first create an array of all neighbours for all valid healsparse pixels
    nside = hp.get_nside(kappa_map) #get nside
    ipix = np.arange(hp.nside2npix(nside))[kappa_map.mask==False][0] #list all pixels and remove masked ones
    #print(ipix.shape)
    neighbours = hp.get_all_neighbours(nside, ipix) #find neighbours for all pixels we care about
    #get kappa values for each pixel in the neighbour array
    neighbour_vals = kappa_map.data[neighbours.T]
    #print(neighbour_vals.shape)
    #get kappa values for all valid healsparse pixels
    pixel_val = kappa_map.data[ipix]
    #compare all valid healsparse pixels with their neighbours to find extrema
    if minima:
        test = np.tile(pixel_val,[8,1]).T 
        #print(test.shape)
        extrema = np.all(np.tile(pixel_val,[8,1]).T < neighbour_vals,axis=-1)
    else:
        extrema = np.all(np.tile(pixel_val,[8,1]).T > neighbour_vals,axis=-1)
        
    #print the number of extrema identified
    #if minima:
        #print(f'number of minima identified: {np.where(extrema)[0].shape[0]}')
    #else:
        #print(f'number of peaks identified: {np.where(extrema)[0].shape[0]}')
    extrema_pos = np.asarray(hp.pix2ang(nside, ipix[extrema],lonlat=lonlat)).T #find the extrema positions
    extrema_amp = kappa_map[ipix][extrema].data #find the extrema amplitudes
    
    return extrema_pos, extrema_amp

def trim_edge(kappa_map, pos, trim_scale, mask_dominated=True, return_mask_boundary=False):
   
    nside = hp.get_nside(kappa_map) #get nside
    ipix = hp.nside2npix(nside)
    ipix_arr = np.arange(ipix) #array of size npix
    
    #code to remove peaks from near mask
    if mask_dominated:
        filt = hp.mask_good(kappa_map.data) #return true where pixel is not masked
        pixel_trim = ipix_arr[filt] #get all pixel indices that are not masked 
        neighbours = hp.get_all_neighbours(nside, pixel_trim) #find neighbours for all unmasked pixels
        #non_pixels_filter = np.where(neighbours == -1) #find all places where pixel has no neighbours 
        
        neighbour_vals = kappa_map.data[neighbours] #get kappa values for each pixel in the neighbour array
        is_masked = np.where( neighbour_vals == hp.UNSEEN ) #find all masked neighbours
        neighbours_masked = neighbours[is_masked] #filter out non masked pixels
        masked_edge_pixels = np.unique( neighbours_masked ) #remove duplicates of same masked pixel
        masked_edge_pixels = np.delete(masked_edge_pixels, np.where(masked_edge_pixels==-1)[0]) #remove values that are -1, these are false pixels
        pixel_ang_trim = np.array( hp.pix2ang(nside, masked_edge_pixels, lonlat=True ) ).T #angular coordinates of pixel boundary of mask
        
        
    if not mask_dominated:
        pixel_trim = ipix_arr[kappa_map.mask] #get all pixel indices that are masked 
        neighbours = hp.get_all_neighbours(nside, pixel_trim) #find neighbours for all pixels we care about
        neighbour_vals = kappa_map.data[neighbours.T] #get kappa values for each pixel in the neighbour array
        bad_val = hp.UNSEEN
        pixel_trim_neighbour = np.any(neighbour_vals != bad_val, axis=-1) #is this line a faster alternative to the following one?
        pixel_ang_trim = np.array( hp.pix2ang( nside, pixel_trim[pixel_trim_neighbour], lonlat=True ) ).T
        
    factor = np.pi / 180.
    btt = BallTree(pixel_ang_trim[:,::-1] * factor, metric='haversine')
    distances, indices = btt.query(pos[:,::-1] * factor, k=1) #[:,::-1] because btt.query takes lat lon and hp uses lon lat.
    mask_filter = distances[:,0] > trim_scale
    
    
    if return_mask_boundary:
        return mask_filter, pixel_ang_trim
    else:
        return mask_filter
    
    
sl_arcmin= 5 #define the smoothing length for the map in arcmins
sl_rad = sl_arcmin/60/180*np.pi #convert sl_arcmin to radians

fid = []
for i in range(200):
    fid.append("{:04d}".format(i)) #index of fiducial cosmology
bins = [0, 1, 2, 3]

for fn in fid:
    for nbin in bins:
        kappa_map_smooth = hp.smoothing(np.load('/project/astro/simulations/CosmoGridConvergence/fiducial_baryonified/nzmap_%s.npy'%fn)[nbin],sigma = sl_rad)

        kappa_masked_smooth = hp.ma(kappa_map_smooth)
        minima_pos, minima_amp = find_extrema(kappa_masked_smooth, minima=True, lonlat=True, trim_edge_scale = 10*sl_rad, mask_dominated = True, return_mask_boundary = False)
        #maxima_pos, maxima_amp = find_extrema(kappa_masked_smooth, minima=False, lonlat=True, trim_edge_scale = 10*sl_rad, mask_dominated = True, return_mask_boundary = False)

        nside = hp.get_nside(kappa_masked_smooth)
        npix = hp.nside2npix(nside)
        minima_pix = hp.ang2pix(nside, minima_pos[:,0], minima_pos[:,1], lonlat = True)
        np.save('/project/ls-mohr/users/xu.han/voids/cosmogrid/fiducial/minima_pix_nzmap_%s_%s_5.npy'%(fn,nbin), minima_pix)

        #first define some useful variables/arrays
        sorted_index = np.argsort(kappa_masked_smooth) #get array that sorts pixels from highest gradient to lowest gradient
        ipix_arr = np.arange(0,npix) #create array of pixel numbers
        nei_pix = hp.get_all_neighbours(nside, ipix_arr).T #create array of pixel neighbours
        no_nei = np.where(nei_pix==-1)
        nei_pix = nei_pix.tolist()
        for i in no_nei[0]:
            nei_pix[i].remove(-1)
        queue = np.full(npix, False) #define priority queue. False if not in queue. True if in queue. np.nan if moved from queue to out of queue.

        #A set of markers, pixels where the flooding shall start, are chosen. Each is given a different label.
        labels_pix = np.zeros((npix,)) #create an array to store pixel labels. 0 counts as not labelled.
        watershed_labels = np.arange(1,len(minima_pix)+1) #create different labels for each minima
        labels_pix[minima_pix] = watershed_labels #assign different labels to each minima pixel

        #The neighboring pixels of each marked area are inserted into a priority queue with a priority level corresponding to the gradient magnitude of the pixel.
        for i in range(npix):
            if labels_pix[i] != 0:        
                for j in nei_pix[i]:
                    queue[j] = True
            
        queue_com = np.dstack((ipix_arr, kappa_masked_smooth, queue))
        queue_com = queue_com[0]
        #print(queue_com)
        queue_com_sorted = queue_com[queue_com[:,1].argsort()]
        ipix_sorted = queue_com_sorted[:, 0].astype(int) #descend ranking of pixel according to mag
        queue_sorted = queue_com_sorted[:, 2].astype(bool) #ranked boolean array of queue
        ipix_com = np.dstack((ipix_arr, ipix_sorted)) #1.ranking of pixel according to mag     2.numbers of pixel
        ipix_com = ipix_com[0]
        ipix_rev = ipix_com[ipix_com[:,1].argsort()][:, 0] #get reversed ranking number of pixel, in order to find the position of boolean value

        #Main
        s = 0
        #queue_cp = copy.copy(queue_sorted)
        queue_cp = copy.copy(queue_sorted)
        labels_pix_cp = copy.copy(labels_pix)
        #The pixel with the highest priority level is extracted from the priority queue. 
        #If the neighbors of the extracted pixel that have already been labeled all have the same label, 
        #then the pixel is labeled with their label. All non-marked neighbors that are not yet in the priority queue are put into the priority queue.
        while np.any(queue_cp[s:]):
            ind = np.argmax(queue_cp[s:])+s
            ind_nei = np.asarray(nei_pix[ipix_sorted[ind]]) #get all labelled neighbours of queried pixel 
            labels_nei = labels_pix_cp[ind_nei]
            ind_nei_add = ind_nei[labels_nei == 0]    #get the index of pixel which should be added to queue
            #add query pixel neighbours without labels to queue
            labels_nei = labels_nei[labels_nei > 0]
            if len(np.unique(labels_nei)) == 1: #if all labels of query pixel neighbours are the same, assign same label to query pixel
                labels_pix_cp[ipix_sorted[ind]] = labels_nei[0]
            else: #if all labels of query pixel neighbours are not the same, assign nan, which is used to represent boundary pixels
                labels_pix_cp[ipix_sorted[ind]] = -1

            queue_cp[ipix_rev[ind_nei_add]] = True
            queue_cp[ind] = False #remove queried pixel from queue
            s += 1
            if s%100 == 0:
                print(s, end='\r')
    
        #Redo step 3 until the priority queue is empty.
        #the while loop should end automatically when the priority queue is empty
        np.save('/project/ls-mohr/users/xu.han/voids/cosmogrid/fiducial/labels_pix_nzmap_%s_%s_5.npy'%(fn,nbin), labels_pix_cp)
        boundary_pix = np.where(labels_pix_cp == -1)
        boundary_pos = hp.pix2ang(nside, boundary_pix, lonlat = True)
        np.save('/project/ls-mohr/users/xu.han/voids/cosmogrid/fiducial/bound_pix_nzmap_%s_%s_5.npy'%(fn,nbin), boundary_pix)

        labels_pix = labels_pix_cp.astype(int)
        boun_nei = hp.get_all_neighbours(nside, boundary_pix[0]).T
        #find boundary of every void
        voids_bound = [[] for i in range(len(minima_pix))]
        num_bound = np.zeros(len(boundary_pix[0]), dtype = int)
        for i in range(len(boundary_pix[0])):
            nei_lab = np.unique(labels_pix[boun_nei[i]].astype(int))
            num_bound[i] = len(nei_lab)-1
            for j in nei_lab:
                if j != -1:
                    voids_bound[j-1].append(boundary_pix[0][i])

        GSN_smooth = hp.read_map('/home/x/Xu.Han/voids/GSN_smooth_512_5.fits')
        sigma_GSN = np.std(GSN_smooth)
        pix_nei = hp.get_all_neighbours(nside, range(npix)).T
        kappa = kappa_masked_smooth

        voids = [np.array([], dtype=int) for i in range(len(minima_amp))]
        for i in range(npix):
            if labels_pix[i] != -1:
                voids[labels_pix[i]-1] = np.append(voids[labels_pix[i]-1], i)
            if i%100==0:
                print(i, end='\r')
        
        np.save('/project/ls-mohr/users/xu.han/voids/cosmogrid/fiducial/voids_nzmap_%s_%s_5.npy'%(fn,nbin), np.array(voids, dtype=object), allow_pickle=True)

        #find labels of neighbouring voids, include self
        nei = [[] for i in range(len(minima_pix))]
        for i in range(len(voids_bound)):
            a = np.unique(hp.get_all_neighbours(nside, voids_bound[i]))
            #remove -1 from neighbouring pixel number
            lab = np.unique(labels_pix[a[a>=0]].astype(int)).tolist()
            lab.remove(-1)
            nei[i] = lab
    
        #find the neighbouring voids which can be merged
        Th = 0.5*sigma_GSN
        mer = [[] for i in range(len(minima_pix))]
        for i in range(len(nei)):
            for j in nei[i]:
                if j>i+1:
                    bound_shared = [k for k,v in Counter(voids_bound[i]+voids_bound[j-1]).items() if v>1]
                    amp_bound = np.mean(kappa[bound_shared])
                    if min(abs(minima_amp[i]-amp_bound),abs(minima_amp[j-1]-amp_bound)) < Th:
                        mer[i].append(j)
                
        label_merge = []
        mer_c = copy.copy(mer)
        for i in range(len(minima_pix)):
            if len(mer[i])>0:
                mer_c[i].append(i+1)
                label_merge.append(mer_c[i])
        
        flat_list = [item for sublist in label_merge for item in sublist]
        repeat = np.unique([x for x in flat_list if flat_list.count(x) > 1])

        while len(repeat)!=0:
            for i in repeat:
                label_merge_c = copy.copy(label_merge)
                c = []
                for j in label_merge_c:
                    if i in j:
                        c.append(j)
                        label_merge.remove(j)
                        #label_merge_c.remove(j)
                if len(c)!=0:
                    new = np.unique([item for sublist in c for item in sublist]).tolist()
                    label_merge.append(new)
            flat_list = [item for sublist in label_merge for item in sublist]
            repeat = np.unique([x for x in flat_list if flat_list.count(x) > 1])
    
        #Find the labels of neighbouring pixels for every pixel, -1 is excluded
        num_nei_lab = [np.array([]) for i in range(npix)]
        for i in range(npix):
            num_nei_lab[i] = np.unique(labels_pix[pix_nei[i]])
            if -1 in num_nei_lab[i]:
                index = np.argwhere(num_nei_lab[i]==-1)
                num_nei_lab[i] = np.delete(num_nei_lab[i], index)
            if i%100==0:
                print(i, end='\r')
        
        #Main
        voids_new = []
        bound_pix_new = copy.copy(boundary_pix[0])
        lab_del = np.unique([item for sublist in label_merge for item in sublist]).tolist() #labels of voids which need to be deleted
        s = 1
        for i in label_merge:
            pix_v = np.array([], dtype=int) #pixel of voids
            pix_bp = np.array([], dtype=int) #potential pixel of boundary
            for j in i:
                pix_v = np.append(pix_v, voids[j-1])
                pix_bp =np.append(pix_bp, np.array(voids_bound[j-1]))
        
            pix_b1 = np.unique(np.array([x for x in pix_bp.tolist() if pix_bp.tolist().count(x) > 1]))
            pix_b2 = copy.copy(pix_b1)
            for k in pix_b1:
                if np.array_equal(np.sort(num_nei_lab[k]),np.sort(i)):
                    index = np.argwhere(bound_pix_new==k)
                    bound_pix_new = np.delete(bound_pix_new, index)           
                else:
                    index = np.argwhere(pix_b2==k)
                    pix_b2 = np.delete(pix_b2, index)
            pix = np.append(pix_v, pix_b2)
            voids_new.append(pix)
            s += 1
            if s%100==0:
                print(s, end='\r')

        lab_del.sort(reverse=True)
        voids_iso = copy.copy(voids)
        for i in lab_del:
            del voids_iso[i-1]
        voids_new.extend(voids_iso)

        np.save('/project/ls-mohr/users/xu.han/voids/cosmogrid/fiducial/voids_merged_nzmap_%s_%s_5.npy'%(fn,nbin), np.array(voids_new, dtype=object), allow_pickle=True)
        np.save('/project/ls-mohr/users/xu.han/voids/cosmogrid/fiducial/bound_pix_merged_nzmap_%s_%s_5.npy'%(fn,nbin), bound_pix_new)
        print('Computing of the fiducial cosmology', fn, 'bin', nbin, 'is completed.', end='\r')