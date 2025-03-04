l = []
while True:
    inp = input("Unesite broj ili Done: ")
    if inp == "Done":
        break

    if not inp.isnumeric():
        print("Unesite broj!")
        continue
    
    l.append(float(inp))

print("Ukupno brojeva: ", len(l))
print("Srednja vrijednost", sum(l)/len(l))
print("Minimalna vrijednost: ", min(l))
print("Maksimalna vrijednost: ", max(l))
print("Sortirana lista: ", sorted(l))