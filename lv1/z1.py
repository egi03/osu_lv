def total_euro(hours, e_per_hour):
    return hours * e_per_hour

hours = int(input("Radni sati: "))
e_per_hour = float(input("eura/e: "))

print("Ukupno: ", total_euro(hours, e_per_hour))