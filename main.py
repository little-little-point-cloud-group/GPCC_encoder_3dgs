from GS_tools import Gaussian


p="1F-geo/"
if __name__ == '__main__':
   g=Gaussian()
   g.copy_to_excel(p+"MPEG151-octree__vs__MPEG151-octree.xlsm",p+"MPEG151-lift.xlsm",1,1)
