from calc import *
while True:
  x1=float(input('enter your first number '))
  x2=float(input('enter your second number '))
  operator=input('enter the operator ')
  if operator=="+":
      result=add(x1,x2)
      print(result)
      final=input('Do you want to continue?y or n ')
      if final=='y':continue
      else:break    
  elif operator=="-":
      result=sub(x1,x2)
      print(result)
      final=input('Do you want to continue?y or n ')
      if final=='y':continue
      else:break
  elif operator=="*":
        result=muiltiply(x1,x2)
        print(result) 
        final=input('Do you want to continue?y or n ')
        if final=='y':continue
        else:break
  elif operator=="/":
        result=div(x1,x2)
        print(result)
        final=input('Do you want to continue?y or n ')
        if final=='y':continue
        else:break
  elif operator=="%":
       result=remainder(x1,x2)
       print(result)
       final=input('Do you want to continue?y or n ')
       if final=='y':continue
       else:break
  elif operator=="**":
        result=power(x1,x2)
        print(result)
        final=input('Do you want to continue?y or n ')
        if final=='y':continue
        else:break
  else:break
        

     
    
