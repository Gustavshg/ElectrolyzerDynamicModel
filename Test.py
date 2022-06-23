from keras.layers import Lambda
square = Lambda(lambda x,y:x-y)
print(square(1,2))