spam = []
ham = []
with open("SMSSpamCollection.txt", encoding="utf8") as f:
    lines = [line.strip() for line in f.readlines()]

    for line in lines:
        if line.startswith("spam"):
            spam.append(line[5:].strip())
        else:
            ham.append(line[4:].strip())


avg_spam = sum([len(s.split(" ")) for s in spam]) / len(spam)
avg_ham = sum([len(h.split(" ")) for h in ham]) / len(ham)
spam_uskl = len([s for s in spam if s[-1] == '!'])

print("Prosjecan broj rijesi u spam poruci: ", avg_spam)
print("Prosjecan broj rijeci u ham poruci: ", avg_ham)
print("Broj poruka spam poruka koje zavrsavaju usklicnikom: ", spam_uskl)