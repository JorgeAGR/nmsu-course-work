binaryInput = str(input('Enter binary code (32 bit): '))

lengthBinary = len(binaryInput)

binaryDigits = []

for i in range(lengthBinary):
    b = int(binaryInput[i])
    binaryDigits.append(b)

decimal = 0

for i in range(9,32):
    if (binaryDigits[i] == True):
        decimal += 2**(i-9)

for i in range(1,9):
    if (binaryDigits[i] ==  True):
        decimal = decimal * 10**(i-1)

if (binaryDigits[i] == True):
    decimal = -decimal

print(decimal)