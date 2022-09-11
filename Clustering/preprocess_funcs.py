import numpy as np

def spheroid(long_sq,short_sq):
    return np.sqrt((long_sq-short_sq)/long_sq) if long_sq!=short_sq else 0
def elip(majorminor):
    maj_sq = majorminor[0]**2
    min_sq = majorminor[1]**2
    return spheroid(maj_sq,min_sq) if maj_sq>min_sq else spheroid(min_sq,maj_sq)
def getAngleRad(diffs):
    return np.arctan2(diffs[1], diffs[0])
def getAngleRadZero(xy):
    return np.arctan2(xy[1]-0, xy[0]-0)

def add_transformations(X_train):
    #ellipticity
    ellipticity = np.apply_along_axis(elip, 2, X_train[:,:,3:5])
    X_train = np.dstack((X_train,ellipticity))
    #step size
    d = np.diff(X_train[:,:,:2],axis=1,prepend=0)
    step_size = np.sqrt(np.power(d,2).sum(axis=2))
    X_train = np.dstack((X_train,step_size))
    #acceleration
    acceleration = np.diff(step_size,axis=1,prepend=0)/30.0
    X_train = np.dstack((X_train,acceleration))
    #angle of step
    d = np.diff(X_train[:,:,:2],axis=1,prepend=0)
    angles_of_step = np.apply_along_axis(getAngleRad, 2, d)
    X_train = np.dstack((X_train,angles_of_step))
    #angle from center
    angles_from_center = np.apply_along_axis(getAngleRad, 2, X_train[:,:,:2])
    X_train = np.dstack((X_train,angles_from_center))

    return X_train
