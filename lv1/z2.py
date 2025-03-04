try:
    grade = float(input("Ocjena: "))
except:
    print("Unesite broj!")
    exit()

if not 0 <= grade <= 1:
    print("Ocjena nije u rasponu!")
    exit()

if grade >= 0.9:
    print("A")
elif grade >= 0.8:
    print("B")
elif grade >= 0.7:
    print("C")
elif grade >= 0.6:
    print("D")
else:
    print("F")