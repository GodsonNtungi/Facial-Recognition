
# imports
# 
from app.modelcreator import create_model

name = input("Enter your name: ")
results = create_model(name)
print(results)