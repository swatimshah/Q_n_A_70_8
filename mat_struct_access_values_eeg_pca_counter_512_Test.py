import scipy.io as sio
import numpy
from numpy import savetxt 
from sklearn.decomposition import PCA
from tensorflow.random import set_seed
from numpy.random import seed

def _check_keys( dict):
	"""
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	"""
	for key in dict:
    		if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
        		dict[key] = _todict(dict[key])
	return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
	"""
	this function should be called instead of direct scipy.io .loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	"""
	data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	return _check_keys(data)


# setting the seed
seed(1)
set_seed(1)

combinedData = numpy.empty([0, 640])

myKeys = loadmat("d:\\a_eeg_1_to_45\\EEGData_unit.mat")
print(myKeys)
eegData = myKeys['EEGData_unit']
eegDataAllSamples = eegData['Data']
eegDataAllLabels = eegData['Labels']
print(eegDataAllSamples.shape)
print(eegDataAllLabels.shape)

yes_input = eegDataAllLabels == 'Yes'
eegDataAllLabels[yes_input] = 1
no_input = eegDataAllLabels == 'No'
eegDataAllLabels[no_input] = 0


for i in range (398):
	eegData = eegDataAllSamples[i].reshape(64, 512)
	eegData = eegData.transpose()
	my_pca = PCA(n_components=10, random_state=2)
	my_pca.fit(eegData)
	print(my_pca.components_.shape)
	combinedData = numpy.append(combinedData, my_pca.components_.flatten().reshape(1, 640), axis=0)

wholeData = numpy.append(combinedData, eegDataAllLabels.reshape(len(eegDataAllLabels), 1), axis=1)

savetxt('d:\\a_eeg_1_to_45\\EEGData_512_Test_New.csv', wholeData, delimiter=',')


