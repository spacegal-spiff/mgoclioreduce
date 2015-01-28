import numpy as np
import scipy as sp
import scipy.ndimage
import pyfits
import sys
import numpy.ma as ma
import math
# import pidly # Not yet... need IDL installation
import os

def make_dark(darkinfilen, targetfilen, outfilen = 'masterdark'):
	"""
	Takes raw dark frames and median combines them to create a masterdark frame.

	Parameters
	----------
	darkinfilen : text file with file names of the dark files you wish to reduce (one per line)
	targetfilen : name of the target 
	
	Returns
	-------
	med_dark : 2D array of the median combination of the dark frames
	(target_date_masterdark.fits) : writes FITS file to disk
	"""

	# step 1: median stack the darks
	darklist = open(darkinfilen,'r')
	darklist = darklist.readlines()
	n = len(darklist)
 
	fits = pyfits.open(darklist[0])
	darkheader = fits[0].header # pull header info from first dark frame
	imdark = fits[0].data
	 
	date = darkheader['DATE']
	date = date[0:10]
	 
	yimsize = np.array(np.shape(imdark))[1]
	ximsize = np.array(np.shape(imdark))[2]
	darkarray = np.zeros((yimsize, ximsize, n), dtype = np.float32)
	 
	print "Creating master dark frame..."
	 
	for i in xrange(0, n):
		fits = pyfits.open(darklist[i])
		im = fits[0].data
		if len(im.shape) == 3: # check for data cubes
			assert not np.any(np.isnan(im))
			im = np.median(im, axis = 0) # if data cube, then take the median of the frames
		darkarray[:, :, i] = im
	 
	med_dark = np.median(darkarray, axis = 2)
	darkname = targetfilen + '_' + date + '_' + outfilen + '.fits'
	pyfits.writeto(darkname, med_dark, darkheader, clobber=True)
	return med_dark



def make_flat(flatinfilen, targetfilen, flatflag, outfilen = 'masterflat'):
	"""
	Takes raw flatfield frames (sky or lamp), dark subtracts them,  median combines them to create a masterdark frame.

	Parameters
	----------
	flatinfilen : text file with file names of the flat files you wish to reduce and combine (one per line)
	targetfilen : name of the target 
	
	Returns
	-------
	med_dark : 2D array of the median combination of the dark frames
	(target_date_masterdark.fits) : writes FITS file to disk
	"""

	

	# Assuming you need to create a single combined flatframe: 
	if flatflag == 1:
		flatlist = open(sys.argv[2], 'r')
		flatlist = flatlist.readlines()
		n = len(flatlist)

		fits = pyfits.open(flatlist[0])
		flatheader = fits[0].header

		imsize = 1024
		flatarray = np.zeros((imsize,imsize,n),dtype=np.float32)

		for i in xrange(0,n):
			fits=pyfits.open(flatlist[i])
			im=fits[0].data
			if len(im.shape) == 3: #check for data cubes
				assert not np.any(np.isnan(im))
				im = np.median(im,axis=0) #if data cube, then median frames
			flatarray[:,:,i]=im

	# if flats are twilight/skyflats:
		if int(sys.argv[4]) == 0:

		# subtract off the median dark frame from each of the twiflats, IF median dark is the same exposure length as the twilight flats. If not, then ???
		for i in xrange (0,n):
			flatarray[:,:,i] -= med_dark
			
			median_flat1 = np.median(flatarray[:,:,0])
			
			for i in xrange(0,n):
				flatmedian=np.median(flatarray[:,:,i])
				flatarray[:,:,i] *= (median_flat1/flatmedian)

				med_flat=np.median(flatarray,axis=2)
				flatname = sys.argv[6] + '_' + date + '_medianflat.fits'
				pyfits.writeto(flatname,med_flat,flatheader,clobber=True)


			else: 
				tmp_medval=np.average(flatarray)
				flaton=np.zeros((imsize,imsize,n),dtype=np.float32)
				flatoff=np.zeros((imsize,imsize,n),dtype=np.float32)
			# changed flaton,off = np.zeros((imsize,imsize,3)) to imsize,imsize,n
			counton = 0
			countoff = 0	

			for i in xrange(0,n):
				if np.median(flatarray[:,:,i]) < tmp_medval:
					flatoff[:,:,countoff] = flatarray[:,:,i]
					countoff += 1
				else:
					flaton[:,:,counton] = flatarray[:,:,i]
					counton += 1

					medflatoff = np.median(flatoff,axis=2)
					medflaton = np.median(flaton,axis=2)	
					med_flat = medflaton - medflatoff
					flatname = sys.argv[6] + '_' + date + '_medianflat.fits'
					pyfits.writeto(flatname,med_flat,flatheader,clobber=True)

					del flatarray
		#del flaton
		#del flatoff
		del darkarray

		ind=np.where(med_flat == 0)
		if np.sum(ind) > 0:	
			med_flat[ind] = 0.001

	elif flatflag == 0:
		calflat = pyfits.open(sys.argv[2])
		calflat = calflat[0].data
		# Strip mode trimming ONLY supported right now (KWD, 2014-11-25)
		if (np.shape(calflat)[0] != yimsize) and (np.shape(calflat)[1] != ximsize):
			calflattrim = calflat[212:512, 0:1024]
			scaled_flat = calflattrim	
	 
	 
	if __name__ == '__main__':
		darkinfile = sys.argv[1]
		target = sys.argv[6] # target name (e.g. HIP100)
		flatflag = sys.argv[8] # whether the flat needs to be constructed or is provided
		make_dark(darkinfile, target)
		make_flat()



'''

#!/Library/Frameworks/Python.framework/Versions/Current/bin/python

import numpy as np
import scipy as sp
import scipy.ndimage
import pyfits
import sys
import numpy.ma as ma
import math
import pidly
import os

idl = pidly.IDL()

# call file as: python reductionscripttest.py darklist flatlist sciencelist 1lamp(or 0 for sky) 0(not saturated) objname alignflag(1 if equal mass binary) flatflag badpixmask.fits

# step 1: median stack the darks
darklist = open(sys.argv[1],'r')
darklist = darklist.readlines()
n = len(darklist)

fits = pyfits.open(darklist[0])
darkheader = fits[0].header
imdark = fits[0].data

date = darkheader['DATE']
date = date[0:10]

yimsize = np.array(np.shape(imdark))[1]
ximsize = np.array(np.shape(imdark))[2]
darkarray = np.zeros((yimsize,ximsize,n),dtype=np.float32)

print "Creating master dark frame..."

for i in xrange(0,n):
	fits = pyfits.open(darklist[i])
	im = fits[0].data
	if len(im.shape) == 3: #check for data cubes
		assert not np.any(np.isnan(im))
		im = np.median(im, axis=0) #if data cube, then median frames
	darkarray[:,:,i]=im

med_dark = np.median(darkarray,axis=2)
darkname = sys.argv[6] + '_' + date + '_masterdark.fits'
pyfits.writeto(darkname, med_dark, darkheader, clobber=True)



# steps 2 and 3: dark subtract and median stack the flats
flatflag = sys.argv[8]

# if you need to create a single combined flatframe: 
if flatflag == 1:
	flatlist = open(sys.argv[2],'r')
	flatlist = flatlist.readlines()
	n=len(flatlist)

	fits=pyfits.open(flatlist[0])
	flatheader=fits[0].header

	imsize=1024
	flatarray=np.zeros((imsize,imsize,n),dtype=np.float32)

	for i in xrange(0,n):
		fits=pyfits.open(flatlist[i])
		im=fits[0].data
		if len(im.shape) == 3: #check for data cubes
			assert not np.any(np.isnan(im))
			im = np.median(im,axis=0) #if data cube, then median frames
		flatarray[:,:,i]=im

# if flats are twilight/skyflats:
	if int(sys.argv[4]) == 0:

	# subtract off the median dark frame from each of the twiflats, IF median dark is the same exposure length as the twilight flats. If not, then ???
	for i in xrange (0,n):
		flatarray[:,:,i] -= med_dark
		
		median_flat1 = np.median(flatarray[:,:,0])
		
		for i in xrange(0,n):
			flatmedian=np.median(flatarray[:,:,i])
			flatarray[:,:,i] *= (median_flat1/flatmedian)

			med_flat=np.median(flatarray,axis=2)
			flatname = sys.argv[6] + '_' + date + '_medianflat.fits'
			pyfits.writeto(flatname,med_flat,flatheader,clobber=True)


		else: 
			tmp_medval=np.average(flatarray)
			flaton=np.zeros((imsize,imsize,n),dtype=np.float32)
			flatoff=np.zeros((imsize,imsize,n),dtype=np.float32)
		# changed flaton,off = np.zeros((imsize,imsize,3)) to imsize,imsize,n
		counton = 0
		countoff = 0	

		for i in xrange(0,n):
			if np.median(flatarray[:,:,i]) < tmp_medval:
				flatoff[:,:,countoff] = flatarray[:,:,i]
				countoff += 1
			else:
				flaton[:,:,counton] = flatarray[:,:,i]
				counton += 1

				medflatoff = np.median(flatoff,axis=2)
				medflaton = np.median(flaton,axis=2)	
				med_flat = medflaton - medflatoff
				flatname = sys.argv[6] + '_' + date + '_medianflat.fits'
				pyfits.writeto(flatname,med_flat,flatheader,clobber=True)

				del flatarray
	#del flaton
	#del flatoff
	del darkarray

	ind=np.where(med_flat == 0)
	if np.sum(ind) > 0:	
		med_flat[ind] = 0.001

elif flatflag == 0:
	calflat = pyfits.open(sys.argv[2])
	calflat = calflat[0].data
	# Strip mode trimming ONLY supported right now (KWD, 2014-11-25)
	if (np.shape(calflat)[0] != yimsize) and (np.shape(calflat)[1] != ximsize):
		calflattrim = calflat[212:512, 0:1024]
		scaled_flat = calflattrim
	

# step 4: create a bad pixel file
#standard deviation of the flat
"""
std = np.std(med_flat[100:924,100:1000])

#some trickery so now that within each pixel is the number of deviations that pixel is away from the median (which will be 1.0 after you scale it)
tmp = abs(med_flat - np.median(med_flat))/std  

#max_std is the number of standard deviations above which a pixel is considered bad. bad will be a list of array indices where the standard deviation is above the max_std.
max_std = 3.0

bad = tmp > max_std

badflat=np.copy(med_flat)
badflat[bad]=0
name = sys.argv[6] + '_' + date +'_badflat.fits'
pyfits.writeto(name,badflat,flatheader,clobber=True)
'''

# step 5: scale median flat by its median value, to get pixel values near 1 -- this is already done for the clio data! will have to revisit for the L' actual flats.
'''
flatscalefactor = np.median(med_flat)

scaled_flat = med_flat/flatscalefactor
name = sys.argv[6] + '_' + date + '_masterflat.fits'
pyfits.writeto(name,scaled_flat,flatheader,clobber=True)

"""
# step 6: subtract master dark from each science frame

scilist = open(sys.argv[3],'r')
scilist = scilist.readlines()
n = len(scilist)

fits = pyfits.open(scilist[0])
sciheader = fits[0].header

angle = np.zeros(n)

yimsize = np.array(np.shape(imdark))[1]
ximsize = np.array(np.shape(imdark))[2]
sciarray = np.zeros((yimsize,ximsize,n),dtype=np.float32)

# hardcoded for now -- this is the value from the 2013 astrometric correction 
northclio = -1.80

for i in xrange(0,n):
	fits = pyfits.open(scilist[i])
	im = fits[0].data
	if len(im.shape) == 3: #check for data cubes
		assert not np.any(np.isnan(im))
		im = np.median(im, axis=0) #if data cube, then median frames
	sciarray[:,:,i] = im
	header = fits[0].header
	# the following is populating the angle array with DEROT values and converting to radians
	angle[i] = (header['ROTOFF'] - 180.0 + northclio)*(np.pi/180.0)

# subtract off the median dark frame from each of the science frames
for i in xrange (0,n):
	sciarray[:,:,i] -= med_dark


# step 7: divide the science frames by the masterflat frame
#### ??? is this right ??? ####
for i in xrange(0,n):
	sciarray[:,:,i] /= scaled_flat

# step 7a .. remove hot pixels from the reduced images # this is unnecessary for the clio data, as the bad pixels are known...
"""

width = 5
sd_cut = 4.0
for i in xrange(0,n):

	#this part is cleaning the bad pixels identified in the flat#
	med_im = sp.ndimage.median_filter(sciarray[:,:,i],width)
	tmp = sciarray[:,:,i]
	tmp[bad] = med_im[bad]
	sciarray[:,:,i]=tmp

	#This part is working out the local variance, and cleaning cosmic rays#
	med_im = sp.ndimage.median_filter(sciarray[:,:,i],width)
	av_im = sp.ndimage.filters.uniform_filter(sciarray[:,:,i],width)
	avsq_im = sp.ndimage.filters.uniform_filter(pow(sciarray[:,:,i],2.0),width)
	var_im = avsq_im - (pow(av_im,2.0))
	sd_im = np.sqrt(var_im)

	ind = np.where(abs((sciarray[:,:,i]-av_im)/sd_im) > sd_cut)
	if np.sum(ind) > 0:
		tmp = sciarray[:,:,i]
		tmp[ind] = med_im[ind]
		sciarray[:,:,i]=tmp

del med_im
del av_im
del avsq_im
del var_im
del sd_im
'''
"""

bpm = pyfits.open(sys.argv[9])
ind = bpm[0].data

for i in xrange(0,n):
	if np.sum(ind) > 0:
		tmp = sciarray[:,:,i]
		tmp[ind] = med_im[ind]
		sciarray[:,:,i] = tmp



# step 8: create master sky frame by median stacking the flattened, dark-subtracted science frames
#If there is a rotation, we want two skies, one for each group (the sky pattern changes when the instrument rotates)
#Note this will only work if there is only two unique rotation values.

rot_flag = 0
if np.sum(np.where(angle != 0.0)) == 0:
	medskyframe=np.median(sciarray,axis=2)
	skyname = sys.argv[6] + '_' + date + '_mastersky.fits'
	pyfits.writeto(skyname,medskyframe,sciheader,clobber=True)
else:
	rot_flag = 1

	ind=np.array(np.where(angle == 0.0))
	medskyframe_a=np.median(sciarray[:,:,ind[0,:]],axis=2)
	ind=np.array(np.where(angle != 0.0))
	medskyframe_b=np.median(sciarray[:,:,ind[0,:]],axis=2)
	skyname = sys.argv[6] + '_' + date + '_masterskyA.fits'
	pyfits.writeto(skyname,medskyframe_a,sciheader,clobber=True)
	skyname = sys.argv[6] + '_' + date + '_masterskyB.fits'
	pyfits.writeto(skyname,medskyframe_b,sciheader,clobber=True)


###################### in progress ######################
# step 9: measure median for each science frame in order to scale each sky

if rot_flag == 0:
	skyfactor = np.median(medskyframe)
else:
	skyfactor_a = np.median(medskyframe_a)
	skyfactor_b = np.median(medskyframe_b)



if rot_flag == 0:
	for i in xrange(0,n):
		scifactor=np.median(sciarray[:,:,i])
		if skyfactor == 0:
			scifactor = 1
			skyfactor = 1
			
		sciarray[:,:,i] -= medskyframe/(skyfactor/scifactor)
		
else:
	ind=np.array(np.where(angle == 0.0)) #Find which frames have no rotation
	ind=ind[0,:]
	for i in xrange(0,len(ind)):
		scifactor=np.median(sciarray[:,:,ind[i]])
		sciarray[:,:,ind[i]] -= medskyframe_a/(skyfactor_a/scifactor)
	ind=np.array(np.where(angle != 0.0)) #Find which frames have some rotation
	ind=ind[0,:]
	for i in xrange(0,len(ind)):
		scifactor=np.median(sciarray[:,:,ind[i]])
		sciarray[:,:,ind[i]] -= medskyframe_b/(skyfactor_b/scifactor)
	


#(measure median for each science frame, put into scalingfactors)

# mastersky median / science frame median = scalingfactor
# mastersky median / scaling factor = science frame median

# step 10: scale the master sky to make equal number of sky frames (one for each science frame)
# for i in range(0,n):
# (divide master sky by i-th scaling factor, put into i-th position in a big array: the scaled sky array)

# step 11: subtract scaled sky array from flattened, dark-sub science frame array to create a reduced science array

# step 12 (I think?): apply bad pixel mask to each of the frames in the reduced science array (moved to step 7a)

#Now we can replace each of the 'bad' pixels with the smoothed average of the surrounding pixels in the original image
xcen = np.zeros(n)
ycen = np.zeros(n)
for i in xrange(0,n):
	print i
	sciname = 'reducedsci_00' + str(i) + '.fits'
	if i >= 10:
		sciname = 'reducedsci_0' + str(i) + '.fits'
	if i >= 100:
		sciname = 'reducedsci_' + str(i) + '.fits'
	pyfits.writeto(sciname,sciarray[:,:,i],sciheader,clobber=True)
	
	idl('name = "'+sciname+'"')
	idl('im=MRDFITS(name,0,/FSCALE,/SILENT)')
	idl('tmp=SMOOTH(im,21,/EDGE_TRUNCATE)')
	idl('tmp[0:100,*]=0.0')
	idl('tmp[924:1023,*]=0.0')
	idl('tmp[*,0:100]=0.0')
	idl('tmp[*,1000:1023]=0.0')
	idl('foo=MAX(tmp,ind,/NAN)')
	idl('ind=ARRAY_INDICES(tmp,ind)')

	if int(sys.argv[7]) == 1:
		idl('ds9')
		idl('!v->im,im')
		idl('!v->imexam,x,y')
		idl('ind[0]=x')
		idl('ind[1]=y')

# add other loop here! for mpfit
	if int(sys.argv[5]) == 0:
		idl('GCNTRD,im[ind[0]-20:ind[0]+20,ind[1]-20:ind[1]+20],20,20,xcen,ycen,3.0')
		idl('xcen += ind[0]-20.0')
		idl('ycen += ind[1]-20.0')
		xcen[i], ycen[i] = idl.xcen, idl.ycen
	else:
		idl('x=ind[0]')
		idl('y=ind[1]')
		idl('sim = im[x-20:x+20,y-20:y+20]')
		idl('weights = (sim*0.0)+1.0')
		idl('weights[WHERE(sim ge 0.7*MAX(sim,/NAN))]=0.0')
		idl('fit=MPFIT2DPEAK(sim,A,WEIGHTS=weights)')
		idl('xcen = A[4]+(x-20.0)')
		idl('ycen = A[5]+(y-20.0)')
		xcen[i], ycen[i] = idl.xcen, idl.ycen

	sciheader.update('CRPIX1A',xcen[i],'primary star X-center')
	sciheader.update('CRPIX2A',ycen[i],'primary star Y-center')
	sciheader.update('CRVAL1A',0,'')
	sciheader.update('CRVAL2A',0,'')
	sciheader.update('CTYPE1A','Xprime','')
	sciheader.update('CTYPE2A','Yprime','')
	sciheader.update('CD1_1A',1,'')
	sciheader.update('CD1_2A',0,'')
	sciheader.update('CD2_1A',0,'')
	sciheader.update('CD2_2A',1,'')
	sciheader.update('BZERO',0,'')
	del sciheader['NAXIS3']

	pyfits.writeto(sciname,sciarray[:,:,i],sciheader,clobber=True)




# step 13: rotate some reduced science frames, if necessary - determine from header?


#The position of the star in the first image
#is used as the reference position

corners = np.zeros([2,4,n])

#xcen, ycen are the list of coordinates for the stars in each image
old_xcen = np.copy(xcen)
old_ycen = np.copy(ycen)

for i in xrange(0,n):
	corners[0,0,i]= 0
	corners[0,1,i]= 0
	corners[0,2,i]= 1024
	corners[0,3,i]= 1024

	corners[1,0,i]= 0
	corners[1,1,i]= 1024
	corners[1,2,i]= 0
	corners[1,3,i]= 1024

	if angle[i] != 0:
#If we need to rotate, do this now
#update both star position, and corner positions

	
#Rotation matrix
		new_x = (np.cos(angle[i])*xcen[i]) + (-(np.sin(angle[i])*ycen[i]))
		new_y = (np.sin(angle[i])*xcen[i]) + (np.cos(angle[i])*ycen[i])	
		
		xcen[i] = new_x
		ycen[i] = new_y

		for j in xrange(0,4):
			new_x = (np.cos(angle[i])*corners[1,j,i]) + (-(np.sin(angle[i])*corners[0,j,i]))
			new_y = (np.sin(angle[i])*corners[1,j,i]) + (np.cos(angle[i])*corners[0,j,i])	
			corners[:,j,i] = [new_y, new_x]

	if i == 0:
		star = [ycen[i], xcen[i]]
	else:
		dx=star[1] - xcen[i]
		dy=star[0] - ycen[i]
		corners[1,:,i]+=dx
		corners[0,:,i]+=dy

#Set so that the image starts at 0,0
dx = np.min(corners[1,:,:])
star[1]-=dx
corners[1,:,:]-=dx

dy = np.min(corners[0,:,:])
star[0]-=dy
corners[0,:,:]-=dy

#and find the maximum size
xsize = np.ceil(np.max(corners[1,:,:]))
ysize = np.ceil(np.max(corners[0,:,:]))

#Restore your original list of star positions within the non-rotated/shifted images
xcen = np.copy(old_xcen)
ycen = np.copy(old_ycen)

# step 14: shift science frames to match reference
big_im = np.zeros((ysize,xsize,n))

for i in xrange(0,n):
	
	xarr=np.array([np.arange(xsize),]*ysize)
	yarr=np.array([np.arange(ysize),]*xsize).transpose()

	if angle[i] != 0.0:
		new_x = (np.cos(angle[i])*xcen[i]) + (-(np.sin(angle[i])*ycen[i]))
		new_y = (np.sin(angle[i])*xcen[i]) + (np.cos(angle[i])*ycen[i])	

		xcen[i] = new_x
		ycen[i] = new_y
	
		xshift = star[1] - xcen[i]
		yshift = star[0] - ycen[i]

		new_x = (np.cos(-angle[i])*xshift) + (-(np.sin(-angle[i])*yshift))
		new_y = (np.sin(-angle[i])*xshift) + (np.cos(-angle[i])*yshift)	
	
		xshift = new_x
		yshift = new_y		
		
		new_x = (np.cos(-angle[i])*xarr) + (-(np.sin(-angle[i])*yarr))
		new_y = (np.sin(-angle[i])*xarr) + (np.cos(-angle[i])*yarr)

		xarr = new_x
		yarr = new_y
		xarr -= xshift
		yarr -= yshift
	else:
		xshift = star[1] - xcen[i]
		yshift = star[0] - ycen[i]
		xarr -= xshift
		yarr -= yshift

	shifted_tmp=sp.ndimage.map_coordinates(sciarray[:,:,i], [yarr.reshape((1,xsize*ysize)), xarr.reshape((1,xsize*ysize))], mode='constant', cval=0.0, order=3)
	shifted_tmp=shifted_tmp.reshape((ysize,xsize))
	shifted_tmp[np.where(shifted_tmp == 0)]=np.nan

	big_im[:,:,i] = shifted_tmp
	print i

	shiftname = sys.argv[6]+'-'+ date + '-NACO-00' + str(i) + '.fits'
	if i >= 10:
		shiftname = sys.argv[6]+'-'+ date + '-NACO-0' + str(i) + '.fits'
	if i >= 100:
		shiftname = sys.argv[6]+'-'+ date + '-NACO-' + str(i) + '.fits'

	sciheader.update('CRPIX1A',star[1],'primary star X-center')
	sciheader.update('CRPIX2A',star[0],'primary star Y-center')
	sciheader.update('CRVAL1A',0,'')
	sciheader.update('CRVAL2A',0,'')
	sciheader.update('CTYPE1A','Xprime','')
	sciheader.update('CTYPE2A','Yprime','')
	sciheader.update('CD1_1A',1,'')
	sciheader.update('CD1_2A',0,'')
	sciheader.update('CD2_1A',0,'')
	sciheader.update('CD2_2A',1,'')
	sciheader.update('BZERO',0,'')

	pyfits.writeto(shiftname,big_im[:,:,i],sciheader,clobber=True)


# step 15: median combine (co-add?) the shifted science frames to produce final image!

#date = header['DATE']
#date = date[0:10]
name = sys.argv[6]+'_'+ date + '_final.fits'

sciheader.update('CRPIX1A',star[1],'primary star X-center')
sciheader.update('CRPIX2A',star[0],'primary star Y-center')
sciheader.update('CRVAL1A',0,'')
sciheader.update('CRVAL2A',0,'')
sciheader.update('CTYPE1A','Xprime','')
sciheader.update('CTYPE2A','Yprime','')
sciheader.update('CD1_1A',1,'')
sciheader.update('CD1_2A',0,'')
sciheader.update('CD2_1A',0,'')
sciheader.update('CD2_2A',1,'')
sciheader.update('BZERO',0,'')

pyfits.writeto('tmp.fits',big_im,sciheader,clobber=True)

del sciarray
del big_im
del shifted_tmp

idl('name = "'+name+'"')
idl('big_im = mrdfits("tmp.fits",0,header,/fscale,/silent)')
idl('big_im = transpose(big_im,[1,2,0])')
idl('mwrfits,big_im,"stack.fits",/create')
idl('med_im = median(big_im,dimension=3)')
idl('sxdelpar,header,"NAXIS3"')
idl('mwrfits,med_im,name,header,/create')

os.remove('tmp.fits')

idl.close()

'''
"""