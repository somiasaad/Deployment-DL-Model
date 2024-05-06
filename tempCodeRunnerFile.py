def make_prediction(path1):
    res = get_predict_feat(path1)
    predictions = loaded_model.predict(res)
    y_pred = np.argmax(predictions)
    return y_pred