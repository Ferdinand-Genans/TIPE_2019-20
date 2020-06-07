import numpy as np
import random
import imageio
import matplotlib.image as mpimg
import algo_ot ### LA ou il y a l'algorithme de sinkhorn et du simplexe

####### Préparation des images ########
def im_mat(I):
    """Converti une image en matrice avec un pixel par ligne"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))

#######################################
def distance(v1,v2):
    """distance euclidienne au carré entre deux vecteurs de R^3 """
    x = (v1[0]-v2[0])*(v1[0]-v2[0])
    y = (v1[1]-v2[1])*(v1[2]-v2[2])
    z = (v1[2]-v2[2])*(v1[2]-v2[2])
    return (x+y+z)

def NearestPoint(vect,Set):
    """Renvoie l'indexe du point de Set qui est le plus proche de vect"""
    mini = 9999
    index = -1
    for i in range(len(Set)):
        dist = distance(vect, Set[i])
        if(dist < mini):
            mini = dist
            index = i
    return index
########## Spécifique K-Nearest Neighbor ########
def KNN(dataSet, neighbors):
    """ Algorithme K-Nearest Neighbors"""
    K = len(dataSet)
    shape = dataSet.shape
    rows = shape[0]
    cols = shape[1]
    tempdataSet = np.zeros((rows, cols))  
    for i in range(K):
        x= dataSet[i]
        index = NearestPoint(x, neighbors)
        tempdataSet[i] = neighbors[index]
    return tempdataSet
def KNearestNeighborTransport(dataSet, neighbors, Transported): 
    '''algorithme K-Nearest Neighbor mais au lieu que cela change chaque point par son voisin le p^lus proche
        on change poar ou le voisin était transporté par l'applicaiton de transport
    '''
    dataSet = np.array(dataSet)
    shape = dataSet.shape
    rows = shape[0]
    cols = shape[1]
    print(shape)
    tempdataSet = np.zeros((rows, cols,3))  
    for i in range(rows):
        print(i)
        for j in range(cols):
            x= dataSet[i][j]
            index = NearestPoint(x, neighbors)
            tempdataSet[i][j] = Transported[index]
    return tempdataSet
########## Spécifique K-Mean #############

###### Séléction des valeurs aléatoires d'un hstogramme, sans répétition #########
def are_equal(v1,v2): #size 3 for each
    ret = True
    for i in range(3):
        ret = v1[i] == v2[i]
        if (ret == False):
            break
    return ret
    
def is_in_vector(value, vect):
    ret = False
    for i in vect:
        ret = are_equal(i, value)
        if (ret == True):
            break
    return ret
        
def convert_to_vect(Lis):
    ret = []
    for i in range(len(Lis)):
        v = [Lis[i][0],Lis[i][1],Lis[i][2]]
        ret.append(v)
    return ret
###########################################

def Mean(Set):
    s =[0,0,0]
    n= len(Set)
    for k in range(n):
        s[0] += Set[0]
        s[1] += Set[1]
        s[2] += Set[2]
    s[0] /=n
    s[1] /=n
    s[2] /=n
    return s
def random_no_duplicate(vect, size):
    length = vect.shape[0]
    ret = []
    while(len(ret)< size):
        n = random.randint(0,length-1)
        value = vect[n]
        if(is_in_vector(value, ret) == False):
            ret.append(value)
    return ret

def KMean(dataSet,K, kmean):
    Means = random_no_duplicate(dataSet,K) 
    if kmean == True:
        print("in")
        for i in range(100):
            clusters = np.zeros(K)
            for j in dataSet:
                index = NearestPoint(j,Means)
                clusters[index]+= j
            for k in range(K):
                Means[k] = Mean(clusters[k])
    return Means
    
###### Transport d'image #########
    
def Transport(Xs, Xt, Matrice, n):
    Transported = []
    for i in range (n):
        for j in range (n):
            scalar = Matrice[i][j] # normally Matrice is a permutaiton matrice so scalar =1  once by line and 0 for the rest
            if(scalar >0):
                Transported.append(Xt[j])
                break
    return Transported
    

def prepare(X1):
    list1D = []
    for a in X1:
        x1 = str(a[0])
        y1 = str(a[1])
        z1 = str(a[2])
        b = '000'
        x = b[0:3-len(x1)]+x1
        y = b[0:3-len(y1)]+y1
        z = b[0:3-len(z1)]+z1
        xyz = x+y+z
        list1D.append(xyz)
    Set = set(list1D)
    SetD = []
    for xyz in Set:
        x = int(xyz[0:3])
        y = int(xyz[3:6])
        z = int(xyz[6:9])
        temp = [x,y,z]
        SetD.append(temp)
    SetD = np.asarray(SetD)
    return SetD

def Cost(M1,M2,K):
    distanceMat = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            distanceMat[i][j] = distance(M1[i],M2[j])
    return distanceMat
    
def transport_couleur(image1,image2, K = 250,regularised = False):
    """ Transporte l'histogramme de couleur de l'image1, sur celuio de l'image2  """
    """ 
        Si kmean = False: les valeurs des centroides sont aléatoires
        Si no_duplicate = True, on selectionnera les moyennes sur l'ensemble des couleurs, sans doublons et non sur l'ensemble des pixels de l'image'
        K: nombres ed couleurs pour interploé les images
        Si regularised = True: on utilise l'algortihme de Sinkhorn, sinon l'algorithme du simplexe
    """
    ####### Travail préalable sur les images #######
    I1 = imageio.imread(image1)
    I2 = imageio.imread(image2)

    X1 = im_mat(I1)
    X2 = im_mat(I2)
    
    S1 = prepare(X1)
    S2 = prepare(X2)    
    
    Means1 = []
    RandIndex = np.random.choice(len(S1),K)
    for RandI in RandIndex:
        Means1.append(S1[RandI])      
    Means2 = []
    RandIndex2 = np.random.choice(len(S2),K)
    for RandI in RandIndex2:
        Means2.append(S2[RandI])
    
    Cout = Cost(Means1,Means2,K)
    
    if regularised == false:
        TransportMat = algo_ot.simplexe([], [], Cout) #lite vide car poids uniformes
    else:
        TransportMat = algo_ot.sinkhorn([],[],Cout,1e-1])
    T = np.asarray(Transport(Means1, Means2, TransportMat, K))
    print(T)
    new = KNearestNeighborTransport(I1,Means1,T)
    new = new.astype('uint8')
    print(new)
    mpimg.imsave("D:/test.png", new)
