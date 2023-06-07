input1 = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    layer_1 = Conv1D(32, kernel_size=3, activation='relu', strides=1)(input1)
    layer_1 = MaxPooling1D(pool_size=2, strides=2)(layer_1)

    layer_1 = SeparableConv1D(32, kernel_size=2, activation='relu', strides=1)(layer_1)
    layer_1 = MaxPooling1D(pool_size=2, strides=2)(layer_1)
    layer_1 = Flatten()(layer_1)

    # 第二个网络模型
    # 输入为32维

    layer_3 = Bidirectional(LSTM(units=32, return_sequences=True))(input1)
    layer_3 = Flatten()(layer_3)

    # 第3个网络模型