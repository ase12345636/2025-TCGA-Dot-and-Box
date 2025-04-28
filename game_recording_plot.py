import os
import matplotlib.pyplot as plt

folder_path = 'game_record\\random\\5x5'
result_first = []
result_second = []
result_total = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            lines = f.readlines()

            tmp1 = lines[-3].strip()
            result_first.append(float(tmp1.lstrip('first:').rstrip('%')))

            tmp2 = lines[-2].strip()
            result_second .append(float(tmp2.lstrip('second:').rstrip('%')))

            tmp3 = lines[-1].strip()
            result_total.append(float(tmp3.rstrip('%')))

plt.figure(figsize=(12, 5))
plt.xlim(0, len(result_first) - 1)
plt.ylim(0, 100)

plt.title("RresNet 73k")
plt.xlabel('Number of Iteration')
plt.ylabel('Winning Rate')

plt.grid(True)
plt.tight_layout()

plt.plot(result_first, linestyle='-', label='First-Move')
plt.plot(result_second, linestyle='-', label='Second-Move')
plt.plot(result_total,  linestyle='-', label='Total')
plt.legend(loc='upper right')

plt.savefig('RresNet_73k.png', dpi=500)
