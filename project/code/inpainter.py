import cv2
import numpy as np

class inpainter:

    def __init__(self, alpha):
        self.image = None
        self.padImage = None
        self.mask = None
        self.padMask = None
        self.offset = None
        self.alpha = alpha
        
        return

    def illegal(self, row : int, col : int) -> bool:
        if(row < 0 or col < 0):
            return True

        if(row >= self.image.shape[0] or col >= self.image.shape[1]):
            return True

        if(self.mask[row, col]): # in hole
            return True
        return False
        

    def initialize_offset(self):
        for i in range(self.mask.shape[0]): # for each row
            for j in range(self.mask.shape[1]): # for each column
                if(not self.mask[i, j]):
                    self.offset[i, j] = [0, 0]
                else:
                    while(1):
                        rRow = np.random.randint(self.mask.shape[0], dtype = np.int32)
                        rCol = np.random.randint(self.mask.shape[1], dtype = np.int32)
                        if(not self.illegal(rRow, rCol)):
                            self.offset[i, j] = [rRow - i, rCol - j]
                            break
        self.offset.astype(np.int32)
        return

    def cropPatch(self, image, i : int, j : int, patchSize : int):
        halfSize = int(patchSize / 2)

        return image[int(i - halfSize) : int(i + halfSize + 1), int(j - halfSize): int(j + halfSize + 1), :]
    
    def patchDiff(self, patch1, patch2):
        if(patch1.shape != patch2.shape):
            return float("inf")
        else:
            diff = (patch1.astype(np.float32) -  patch2.astype(np.float32))
        
        return np.linalg.norm(diff)

    def smallestPatchDiff(self, diff1, diff2, diff3):
        if(diff1 <= diff2):
            if(diff1 <= diff3):
                return 1
            else:
                return 3
        else:
            if(diff2 <= diff3):
                return 2
            else:
                return 3

    def propagation(self, count : int, row : int, col : int, patchSize: int):
        padSize = patchSize // 2
        if(count % 2 == 0):         

            holePatch = self.cropPatch(self.padImage, row+padSize, col+padSize, patchSize)

            refPos = (self.offset[row, col] + np.array([row, col])).astype(np.int32)
            refPatch = self.cropPatch(self.padImage, refPos[0]+padSize, refPos[1]+padSize, patchSize)

            lrefPos =(self.offset[row, col-1] + np.array([row, col-1])).astype(np.int32)
            lrefPos[1] += 1
            lrefPatch = self.cropPatch(self.padImage, lrefPos[0]+padSize, lrefPos[1]+padSize, patchSize)

            urefPos =(self.offset[row-1, col] + np.array([row-1, col])).astype(np.int32)
            urefPos [0] += 1
            urefPatch = self.cropPatch(self.padImage, urefPos[0]+padSize, urefPos[1]+padSize, patchSize)

            refDiff = self.patchDiff(holePatch, refPatch)
            lrefDiff = self.patchDiff(holePatch, lrefPatch)
            urefDiff = self.patchDiff(holePatch, urefPatch)
            if(self.illegal(lrefPos[0], lrefPos[1])):
                lrefDiff = float("inf")
            if(self.illegal(urefPos[0], urefPos[1])):
                urefDiff = float("inf")

            smallIdx = self.smallestPatchDiff(refDiff, lrefDiff, urefDiff)
            #print(f"        Small Idx : ", smallIdx)
            if(smallIdx == 2):
                self.offset[row, col] = self.offset[row, col-1]
            elif(smallIdx == 3):
                self.offset[row, col] = self.offset[row-1, col]
                    
        else:
        
            holePatch = self.cropPatch(self.padImage, row+padSize, col+padSize, patchSize)

            refPos = (self.offset[row, col] + np.array([row, col])).astype(np.int32)
            refPatch = self.cropPatch(self.padImage, refPos[0]+padSize, refPos[1]+padSize, patchSize)

            rrefPos =(self.offset[row, col+1] + np.array([row, col+1])).astype(np.int32)
            rrefPos[1] -= 1
            rrefPatch = self.cropPatch(self.padImage, rrefPos[0]+padSize, rrefPos[1]+padSize, patchSize)

            brefPos = (self.offset[row+1, col] + np.array([row+1, col])).astype(np.int32)
            brefPos [0] -= 1
            brefPatch = self.cropPatch(self.padImage, brefPos[0]+padSize, brefPos[1]+padSize, patchSize)

            refDiff = self.patchDiff(holePatch, refPatch)
            rrefDiff = self.patchDiff(holePatch, rrefPatch)
            brefDiff = self.patchDiff(holePatch, brefPatch)
            if(self.illegal(rrefPos[0], rrefPos[1])):
                rrefDiff = float("inf")
            if(self.illegal(brefPos[0], brefPos[1])):
                brefDiff = float("inf")

            smallIdx = self.smallestPatchDiff(refDiff, rrefDiff, brefDiff)
            #print(f"        Small Idx : ", smallIdx)
            if(smallIdx == 2):
                self.offset[row, col] = self.offset[row, col+1]
            elif(smallIdx == 3):
                self.offset[row, col] = self.offset[row+1, col]
        return

    def randomSearch(self, row : int, col : int, patchSize: int):
        iterCount = 0.
        padSize = patchSize // 2
        w = max(self.image.shape)
        count = 0
        while(1):
            count+= 1
            if(count > 50):
                break

            randomRow = np.random.rand(1)
            randomCol = np.random.rand(1)
            
            fraction = w * np.power(self.alpha, iterCount)
            #if(count == 1):
                #print(self.offset[row, col], fraction)
            if(fraction < 1.):
                break


            newRowOffset = int(self.offset[row, col][0] +  fraction * randomRow)
            newColOffset = int(self.offset[row, col][1] +  fraction * randomCol)

            newRefRow = newRowOffset + row
            newRefCol = newColOffset + col

            if(self.illegal(newRefRow, newRefCol)):

                continue

            count = 0

            holePatch = self.cropPatch(self.padImage, row + padSize, col + padSize, patchSize)

            oriPatch = self.cropPatch(self.padImage, row + self.offset[row, col][0] + padSize, col + self.offset[row, col][1] + padSize, patchSize)
            newPatch = self.cropPatch(self.padImage, newRefRow + padSize, newRefCol + padSize, patchSize)

            oldDiff = self.patchDiff(holePatch, oriPatch)
            newDiff = self.patchDiff(holePatch, newPatch)

            if(oldDiff > newDiff):
                self.offset[row, col][0] = newRowOffset
                self.offset[row, col][1] = newColOffset
            iterCount += 1
        return

    def pastePatch(self,  patchSize):
        padSize = int(patchSize / 2)
        halfSize = padSize
        for i in range(0, self.mask.shape[0]):
            for j in range(0, self.mask.shape[1]):
            
                if(self.mask[i, j] == 1):
                    #print(f"Paste {i}, {j}")
                    refRow = int(i + self.offset[i, j][0])
                    refCol = int(j + self.offset[i, j][1])
                    #print(f"    with {refRow}, {refCol}")
                    refPatch = self.cropPatch(self.padImage, refRow+padSize, refCol+padSize, patchSize)
                    patchMask = self.padMask[i - halfSize: i + halfSize+1, j - halfSize : j + halfSize+1] 
                    patchMask = np.dstack((patchMask, patchMask, patchMask))
                    diff =  refPatch - self.padImage[i - halfSize: i + halfSize+1, j - halfSize : j + halfSize+1]

                    self.padImage[i - halfSize: i + halfSize+1, j - halfSize : j + halfSize+1] += (diff * patchMask)
        self.image = self.padImage[halfSize: -halfSize, halfSize:-halfSize]

        return

    def inpaint(self, image, mask, patchSize):
        self.image = image
        # print(self.image.shape)
        self.mask = mask
        self.offset = np.zeros(self.mask.shape + tuple([2]))
        self.initialize_offset()
        
        padSize = int(patchSize / 2)
        self.padMask = np.pad(self.mask, ((padSize, padSize), (padSize, padSize)), 'symmetric')
        self.padImage = np.pad(self.image, ((padSize, padSize), (padSize, padSize), (0, 0)), 'symmetric') #pad for dealing with margin
        
        for t in range(40):

            print(f"Iteration {t} : ")
            
            if(t % 2 == 0):
                for i in range(self.mask.shape[0]): # for each row
                    for j in range(self.mask.shape[1]): # for each column
                        if(self.mask[i, j]):
                            #print(f"    Prop {i}, {j}")
                            self.propagation(t, i, j, patchSize)
                            #print(f"    Search {i}, {j}")
                            self.randomSearch(i, j, patchSize)
            else:
                for i in range(self.mask.shape[0])[::-1]: # for each row
                    for j in range(self.mask.shape[1])[::-1]: # for each column
                        if(self.mask[i, j]):
                            self.propagation(t, i, j, patchSize)
                            self.randomSearch(i, j, patchSize)
            self.pastePatch(patchSize)
            # cv2.imshow("Paste", self.image)
            # cv2.waitKey(1000)

        return self.image


    