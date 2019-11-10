#import tensorflowjs as tfjs

# textgen = textgenrnn()
# textgen.reset()
# textgen.train_from_file('./data/s1.txt', num_epochs=60)
# textgen.generate()


def getQuote(pref=""):
    from textgenrnn import textgenrnn
    textgen_2 = textgenrnn('textgenrnn_weights.hdf5')
    generated_texts = textgen_2.generate(n=1, prefix=pref, return_as_list=True)
    print(generated_texts)
    #tfjs.converters.save_keras_model('textgenrnn_weights.hdf5', '')
    return generated_texts

# getQuote()