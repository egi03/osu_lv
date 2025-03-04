with open("song.txt") as f:
    words = [word.rstrip(",").lower() for line in f.readlines() for word in line.strip().split(" ")]

hmap = {}
for w in words:
    if w not in hmap:
        hmap[w] = 1
    else:
        hmap[w] += 1

single = {k for k, v in hmap.items() if v == 1}
print("Rijeci koje se pojavljuju samo jednom: \n", "\n ".join(single))
print("Ukupno riječi koje se pojavljuju samo jednom: ", len(single))