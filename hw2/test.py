# encoding: utf-8

def extendList(val,list=[]):
	list.append(val)
	return list

list1=extendList(10)
print list1
list2=extendList(123,[])
print list2
list3=extendList('a')
print list3
