import matplotlib.pyplot as plt

filename = "Datas/debt.txt"

def read_debt_filename():
    with open(filename, "r") as f:
        debt = f.read()
    return debt

def getData():
    debt = read_debt_filename()
    debt_dict = {}
    list_debt = debt.split("\n")
    name_column = list_debt[0][2:].split(", ")
    len_values = len(name_column)
    for i in range(len_values):    
        debt_dict[name_column[i]] = []
    for line in list_debt[3:]:
        if line == "": continue
        values = line.split(",")
        for i in range(len_values):
            if values[i] == "": debt_dict[name_column[i]].append(None)
            elif i == 2:
                debt_dict[name_column[i]].append(float(values[i]))
            else:
                debt_dict[name_column[i]].append(int(values[i]))
    return debt_dict

data = getData()
years = data['year']
debt_million = data['debt in million euros']
debt_gdp = data['debt / GDP %']

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(years, debt_gdp, color='blue', marker='o', label="Debt / GDP [%]")
ax1.set_xlabel('Year')
ax1.set_ylabel('Debt / GDP [%]', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(years, debt_million, color='red', marker='s', label="Debt [Million Euros]")
ax2.set_ylabel('Debt [Million Euros]', color='red')
ax2.tick_params(axis='y', labelcolor='red')
# ax2.set_yscale('log')

fig.suptitle('National Debt Overview', fontsize=14)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

ax1.grid(color='lightgray', linestyle='--', linewidth=0.7)
ax2.grid(color='lightgray', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.savefig("Figures/debt_analysis.pdf")
# plt.show()