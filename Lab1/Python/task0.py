from PIL import Image

im = Image.open("test.jpg")

print("---Numbers and other data types----")

print(type(-75))
print(type(5.0))
print(type(12345678901))

print("---Strings----")

print(" This is a string ")
print("This is a string, too")
print(type("This is a string "))

print("---Lists and tuples----")

print([1, 3, 4, 1, 6])
print(type([1, 3, 4, 1, 6]))
print(type((1, 3, 2)))

print("--- The range function----")

print(range(17))
print(range(1, 10))
print(range(-6, 0))
print(range(1, 10, 2))
print(range(10, 0, -2))

print("---Operators----")

x = 2 + 2
print(x)
x = 380.5

print(x)
y = 2 * x
print(y)

print("---DECISIONS----")

x = 1
if x > 0:
    print(" Friday is wonderful")
else:
    print(" Monday so perform LAB")
print(" Have a good weekend")

print("---LOOPS--FOR--")

for i in [2, 4, 6, 0]:
    print(i)

print("---LOOPS--while--")
n = 0
while n < 10:
    print(n)
    n = n + 3
