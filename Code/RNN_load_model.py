model.load('trained_models_pretrained_word_embeddings/trained_model_19.h5')
model.set_models(inference=True)
model.set_prevent_update(True)

start = np.random.randint(0, len(X)-1)
pattern = list(X[start])

print("Input:")
print('\'',' '.join(tokenizer.inverse_transform(pattern)),'\'')

sentence_list=[]
print('\n\nPlease wait!! generating sentences...')
for i in range(100):
    for t in range(look_back):
        pred = rm.softmax(model(pattern[t].reshape(1,1))).as_ndarray()
        pred = np.argmax(pred)
    model.truncate()
    pattern = np.delete(pattern, 0)
    pattern = np.append(pattern, pred)
    sentence_list.append(pred)

print('\n\nOutput:')
print('\'',' '.join(tokenizer.inverse_transform(sentence_list)),'\'')

