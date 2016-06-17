
## Choose which list you want to sort

#a = [3,2,5,10,38,47,92,1,3,5,23,12]
a = ["Jimi","vick","Alan"]
# sort the list
a.sort()
print(a)

#b = [6,2,32,34,35,68,73,1,34,84,15]
b = ["A","Z","m"]
# sort the list
b.sort()
print(b)

def mergeSortedLists(a,b):
    l = []
    while a and b:
        if a[0] < b[0] :
            l.append(a.pop(0))
        else:
            l.append(b.pop(0))
    return l + a + b

result = mergeSortedLists(a,b) 
print(result)  
