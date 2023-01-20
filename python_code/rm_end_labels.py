
a_file = open("vysledky/labWithEnd/PD_F_TSK7_withEnd.lab", "r")
lines = a_file.readlines()
a_file.close()

new_file = open("vysledky/labWithEnd/PD_F_TSK7.lab", "w+")
for line in lines:
    if line[-4:] != "end\n":
        new_file.write(line)
new_file.close()
