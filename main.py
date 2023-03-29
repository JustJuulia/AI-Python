import os
import datetime
with open("number.txt", "r") as f:
    current_number = float(f.readline())
money = float(input("What is the amount of money that you are trying to save?: "))
operation = input(f"The current number is {current_number}. Do you want to add or subtract? (Enter + or -): ")
amount = float(input(f"How much do you want to {operation}? "))
if operation == "+":
    current_number += amount
else:
    current_number -= amount
diff = money - current_number
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open("log.txt", "a") as f:
    f.write(f"{timestamp}: {operation} {amount}\n")
with open("number.txt", "w") as f:
    f.write(str(current_number))
last_modified = os.path.getmtime("number.txt")
last_modified_str = datetime.datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S")
print(f"The new number is {current_number}. You need {diff} more to reach {money}.")
print(f"The file was last modified on {last_modified_str}.")