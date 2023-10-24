# Gets user input

# scegli dati
data_types = float(input("What type of data do you have?\n 1 - Unlabeled \t 2 - Labeled \t 3 - Time-lagged --> "))

# cegli cv
if data_types == 1:
    cv_model = input("For unlabeled data you can use\n 1 - Autoencoder CV \t 2 - Variational Autoencoder CV --> ")
if data_types == 2:
    cv_model = float(input("For labeled data you can use\n 1 - Deep-LDA CV \t 2 - Deep-TDA CV --> "))


# chiama il giusto file/funzione
if cv_model == 2:
    print('Complimenti! Stai usando la CV migliore al mondo!')

# inzializza cv


# Uses user input to print out information
# print("Hello " + name + "!")
# print(str(num) + "?! That's my favorite number too!")
# print("")

# # Prints out the variable type
# print("The variable 'name' is a: ")
# print(type(name))
# print("The variable 'num' is a: ")
# print(type(num))