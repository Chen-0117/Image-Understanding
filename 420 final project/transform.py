import os
path='thomas kretschmann'       

#get all files, store in a list
fileList=os.listdir(path)
n=20
for i in fileList:
    
    oldname=path+ os.sep + fileList[n-20]   
    newname=path + os.sep + path + '.2.'+str(n+1)+'.jpeg'
    
    os.rename(oldname,newname)  
    print(oldname,'======>',newname)
    
    n+=1