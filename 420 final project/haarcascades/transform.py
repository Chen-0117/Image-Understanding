import os
path='COnstance Wu'
#get all files, store in a list
fileList=os.listdir(path)
n=0
for i in fileList:
    
    oldname=path+ os.sep + fileList[n]   
    newname=path + os.sep + path + '.6.'+str(n+1)+'.jpeg'
    
    os.rename(oldname,newname)  
    print(oldname,'======>',newname)
    
    n+=1