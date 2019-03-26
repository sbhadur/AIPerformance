
def pred(model, message):
    if (model.predict(message) == 1):
        print("FUNNY")
    else:
        print("NOT FUNNY")
