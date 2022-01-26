%matplotlib inline 
import numpy as np
from matplotlib import pylab as plt
import cv2

cubeAspect=0.4
dotAspect=0.4

class RRect:
    def __init__(self, p0, s, ang):
        self.p0 = (int(p0[0]),int(p0[1]))
        (self.width, self.height) = s
        self.angle = ang
        self.v0, self.v1,self.v2,self.v3 = self.get_verts(p0,s[0],s[1],ang)
        self.verts = [self.v0,self.v1,self.v2,self.v3]
        self.v0s, self.v1s,self.v2s,self.v3s = self.get_verts(p0,s[0],s[1],0)
        self.verts_straight = [self.v0s,self.v1s,self.v2s,self.v3s]
        self.area = int(self.width*self.height)
        if(self.height ==0 or self.width==0):
            self.aspect='undefined'
        else:
            self.aspect=abs(self.width/self.height -1)        
    def get_verts(self, p0, W, H, ang): #p0 - wspol srodka, W -width, H - height, ang - kat
        sin = np.sin(ang/180*3.14159)
        cos = np.cos(ang/180*3.14159)
        P0 = [p0[0]+(W/2*cos)-(H/2*sin), p0[1]+(W/2*sin)+(H/2*cos)] # top right
        P1 = [p0[0]-(W/2*cos)-(H/2*sin), p0[1]-(W/2*sin)+(H/2*cos)] # top left
        P2 = [p0[0]-(W/2*cos)+(H/2*sin), p0[1]-(W/2*sin)-(H/2*cos)] # bot left
        P3 = [p0[0]+(W/2*cos)+(H/2*sin), p0[1]+(W/2*sin)-(H/2*cos)] # bot right
        return [P0,P1,P2,P3]
        
def tupToInt(tup):
    temp = list(tup)
    for i in range(len(temp)):
        temp[i] = int(temp[i])
    return tuple(temp)

def listSubs(list1, list2):
    temp = []
    for i in range(len(list1)):
        temp.append(list1[i]-list2[i])
    return temp


def distBetween(list1, list2):
    listSubbed = listSubs(list1,list2)
    suma = sum([a*a for a in listSubbed])
    dist = abs(suma)**(1/2)
    return dist


def searchForCubeSize(contours_):
    areas =[]
    for i in contours_:
        minar = cv2.minAreaRect(i)
        newRec = RRect(list(minar[0]), list(minar[1]), minar[2])
        if(newRec.width!=0 and newRec.height!=0 and newRec.aspect<cubeAspect):
            areas.append(newRec.area)
    maximal = max(areas)
    areas_1, areas_2 =[], []
    for i in areas:
        if i > 0.4*maximal:
            areas_1.append(i)
    avarage = sum(areas_1)/len(areas_1)
    for i in areas:
        if i > 0.6*avarage:
            areas_2.append(i)
    return min(areas_2), max(areas_1)

def searchForDotSize(contours_):
    areas =[]
    for i in contours_:
        minar = cv2.minAreaRect(i)
        newRec = RRect(list(minar[0]), list(minar[1]), minar[2])
        if(newRec.width!=0 and newRec.height!=0 and newRec.aspect<dotAspect):
            areas.append(newRec.area)
    maximal = max(areas)
    areas_1, areas_2 =[], []
    for i in areas:
        if i > 0.4*maximal:
            areas_1.append(i)
    avarage = sum(areas_1)/len(areas_1)
    for i in areas:
        if i > 0.6*avarage:
            areas_2.append(i)
    return min(areas_2), max(areas_1)


def main(image_name, normalize=True):
    fontScale = 5
    thickness=5
    kernel = np.ones((7,7), np.uint8)
    image = cv2.imread(image_name)
    #image_name_ = image_name.split('/')[1].split('.')[0]
    if normalize:
        image = cv2.resize(image, (4032,1816))
    #cv2.imwrite('trudnedosprawka/'+image_name_+'in.jpg', image)
    image_cpy1 = image.copy()
    image_cpy2 = image.copy()
    image_cpy1 = cv2.convertScaleAbs(image_cpy1, alpha=0.9, beta=10)

    grayImage1 = cv2.cvtColor(image_cpy1, cv2.COLOR_BGR2GRAY)

    grayImage1 = cv2.GaussianBlur(grayImage1, (3,3), cv2.BORDER_ISOLATED)

    _, grayImage1 =  cv2.threshold(grayImage1, 170, 255, cv2.THRESH_BINARY)

    grayImage1 = cv2.Canny(grayImage1, 100, 200, apertureSize = 7, L2gradient = True)

    grayImage1 = cv2.dilate(grayImage1, kernel, iterations=2)

    contours1, _  = cv2.findContours(grayImage1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    grayImage1 = cv2.erode(grayImage1, kernel, iterations=1)
    contours2, _  = cv2.findContours(grayImage1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    image_cpy2 = cv2.convertScaleAbs(image_cpy2, alpha=0.9, beta=0)
    grayImage2 = cv2.cvtColor(image_cpy2, cv2.COLOR_BGR2GRAY)
    grayImage2 = cv2.GaussianBlur(grayImage2, (3,3), cv2.BORDER_DEFAULT)

    _, grayImage2 =  cv2.threshold(grayImage2, 140, 255, cv2.THRESH_BINARY)
    grayImage2 = cv2.Canny(grayImage2, 80, 200, apertureSize = 7, L2gradient = True)
    contours3, _  = cv2.findContours(grayImage2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = contours1+contours2+contours3
    contours = np.array(sorted(contours, key=len, reverse=False), dtype=object)

    diceRects=[]
    botPer, topPer=searchForCubeSize(contours)
    for c in contours:
        minar = cv2.minAreaRect(c)
        newRec = RRect(list(minar[0]), list(minar[1]), minar[2])
        if(newRec.aspect!='undefined'):
            if(newRec.aspect < cubeAspect and newRec.area >= botPer and newRec.area <= topPer ):
                process = 1
                for j in diceRects:
                    dist = distBetween(list(newRec.p0),list(j.p0))
                    if(dist<50):
                        process = 0
                        break
                if(process==1):
                    diceRects.append(newRec)
                    cv2.line(image, tupToInt(newRec.verts[0]), tupToInt(newRec.verts[1]), [0, 255, 0], 3*fontScale, cv2.LINE_AA)
                    cv2.line(image, tupToInt(newRec.verts[1]), tupToInt(newRec.verts[2]), [0, 255, 0], 3*fontScale, cv2.LINE_AA)
                    cv2.line(image, tupToInt(newRec.verts[2]), tupToInt(newRec.verts[3]), [0, 255, 0], 3*fontScale, cv2.LINE_AA)
                    cv2.line(image, tupToInt(newRec.verts[3]), tupToInt(newRec.verts[0]), [0, 255, 0], 3*fontScale, cv2.LINE_AA)
    cv2.putText(image, 'Kosci: '+str(len(diceRects)), (20,250), cv2.FONT_HERSHEY_DUPLEX, 2.0*fontScale, [0, 0, 255], thickness, cv2.LINE_AA)
    dotsSum =0
    for dice in diceRects:
        rot_mat = cv2.getRotationMatrix2D(dice.p0, dice.angle, 1.0)  
        srcTri = np.float32([dice.v0, dice.v1, dice.v2])
        dstTri = np.float32([dice.v0s, dice.v1s, dice.v2s])   
        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        warp_dst = cv2.warpAffine(grayImage1, warp_mat, (grayImage1.shape[1], grayImage1.shape[0]))
        warp_rotate_dst = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
        
        diceImg1 = cv2.getRectSubPix(warp_rotate_dst, (int(dice.width -10), int(dice.height-10)), dice.p0)
        diceImg2 = diceImg1.copy()
        diceImg3 = diceImg1.copy()
        diceImg4 = diceImg1.copy()
        diceImg5 = diceImg1.copy()
        
        _, diceImg1=cv2.threshold(diceImg1, 0,100,cv2.THRESH_BINARY)
        diceContours1, _ = cv2.findContours(diceImg1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        _, diceImg2=cv2.threshold(diceImg2, 0,150,cv2.THRESH_BINARY)
        diceContours2, _ = cv2.findContours(diceImg2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        _, diceImg3=cv2.threshold(diceImg3, 0,200,cv2.THRESH_BINARY)
        diceContours3, _ = cv2.findContours(diceImg3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        _, diceImg4=cv2.threshold(diceImg4, 0,200,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        diceContours4, _ = cv2.findContours(diceImg4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        _, diceImg5=cv2.threshold(diceImg5, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        diceImg5 = cv2.Canny(diceImg5, 80, 200, apertureSize = 7, L2gradient = True)
        diceContours5, _ = cv2.findContours(diceImg5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        diceContours = diceContours1+diceContours2+diceContours3+diceContours4+diceContours5
        diceContours = np.array(sorted(diceContours, key=len, reverse=False), dtype=object)
        
        dotsRect=[]
        botPerDot, topPerDot = searchForDotSize(diceContours)
        for diceContour in diceContours:
            tempRect = cv2.minAreaRect(diceContour)
            newDotRec = RRect(list(tempRect[0]), list(tempRect[1]), tempRect[2])
            if(newDotRec.aspect!='undefined'):
                if(newDotRec.aspect<dotAspect and newDotRec.area>=botPerDot and newDotRec.area <= topPerDot):
                    process = 1
                    for j in range(len(dotsRect)):
                        dist = distBetween(list(newDotRec.p0),list(dotsRect[j].p0))
                        if(dist<10):
                            process = 0
                            break
                    if(process==1):
                        dotsRect.append(newDotRec)        
        cv2.putText(image, str(len(dotsRect)), (dice.p0[0], dice.p0[1]), cv2.FONT_HERSHEY_DUPLEX, 1.75*fontScale, [0, 0, 255], thickness, cv2.LINE_AA)       
        dotsSum+=len(dotsRect)
    image = cv2.putText(image, 'Oczek: '+str(dotsSum), (2400,250), cv2.FONT_HERSHEY_DUPLEX, 2.0*fontScale, [0, 0, 255], thickness, cv2.LINE_AA)
    plt.imshow(image)
    #cv2.imwrite('trudnedosprawka/'+image_name_+'out.jpg', image)

