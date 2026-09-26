# Text Generation with LSTM & GRU

A character-level and word-level text generation project built with TensorFlow/Keras. The notebook trains recurrent neural networks (LSTM and GRU) on text from Project Gutenberg (*Alice's Adventures in Wonderland* by default) and compares their performance at generating new text.

## Overview

The notebook walks through three stages:

1. **Character-level LSTM** — trains an LSTM to predict the next character in a sequence, then generates text at different "temperatures" (randomness levels).
2. **Word-level LSTM** — tokenizes text at the word level and trains a deeper LSTM (with dropout) to predict the next word given a sequence of 30 previous words.
3. **Word-level GRU** — trains a GRU model on the same word-level data as a comparison baseline against the LSTM.

The final section compares the LSTM and GRU word-level models side by side: training/validation loss and accuracy curves, final metrics, and sample generated text from both.

## Requirements

```bash
pip install tensorflow numpy matplotlib
```

Tested with TensorFlow 2.x. A GPU is recommended for training but not required (training will just be slower on CPU).

## How it works

### 1. Data
Text is downloaded automatically from [Project Gutenberg](https://www.gutenberg.org/) and cleaned (headers/footers stripped, lowercased, extra whitespace collapsed) into a local `training.txt` file.

### 2. Character-level model
- Text is converted into integer-encoded sequences of 100 characters.
- Model: `Embedding → LSTM(512) → Dense(vocab_size)`
- Trained with sparse categorical cross-entropy.
- Text is generated one character at a time, sampled with a configurable temperature.

### 3. Word-level models
- Text is tokenized into words (vocabulary capped at 15,000 words) and split into sequences of 30 words predicting the next word.
- **LSTM model**: `Embedding → LSTM(128) → Dropout → LSTM(128) → Dropout → Dense(128, relu) → Dense(word_vocab_size, softmax)`
- **GRU model**: same architecture with `GRU` layers in place of `LSTM`, for direct comparison.
- Both are trained with early stopping on validation loss.

### 4. Comparison
The last section plots both models' loss/accuracy curves together, prints a final metrics table, and generates text from the same seed phrase with each model so you can compare fluency and coherence.

## Usage

Run the notebook top to bottom in Jupyter or Google Colab:

```bash
jupyter notebook earthquake_fixed.ipynb
```

To generate text with a trained model:

```python
# Character-level
generate_text(model, start_string="alice was ", num_generate=500, temperature=0.8)

# Word-level
generate_word_text(word_model, word_tokenizer, "alice was beginning to", num_words=50, temperature=0.7)
```

Lower temperatures (e.g. 0.3) produce safer, more repetitive text; higher temperatures (e.g. 1.2) produce more varied but less coherent text.

## Notes

- Swap in any other Project Gutenberg text by changing the `url` in `create_training_dataset()`.
- The `SEQ_LENGTH` (char-level) and `SEQUENCE_LENGTH` (word-level) constants control how much context the model sees before predicting the next token.
- Training time scales with `EPOCHS`, dataset size, and model size — reduce `rnn_units`/`LSTM` units for faster iteration on CPU.

## License

Add a license of your choice (e.g. MIT) here.
