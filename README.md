Beam search for neural network sequence to sequence (encoder-decoder) models.


Usage example:

```python
from beam_search import beam_search

# Load model and vocabularies...
input_text = "Hello World !"
X = [encoder_vocabulary.get(t, encoder_vocabulary['<UNK>']) for t in input_text.split()]

hypotheses = beam_search(model.initial_state_function, model.generate_function, X, decoder_vocabulary['<S>'], decoder_vocabulary['</S>'])

for hypothesis in hypotheses:

    generated_indices = hypothesis.to_sequence_of_values()
    generated_tokens = [reverse_decoder_vocabulary[i] for i in generated_indices]

    print(" ".join(generated_tokens))`
```