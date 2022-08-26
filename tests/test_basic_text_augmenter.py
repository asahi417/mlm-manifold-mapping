from m3.basic_text_augmenter import text_augmenter

sample = "TextAttack is a Python framework for adversarial attacks, data augmentation, and model training in NLP."
text_augmenter(sample, 'back_translation', 3)
text_augmenter(sample, 'eda', 3)
text_augmenter(sample, 'word_swapping_synonym', 3)
text_augmenter(sample, 'word_swapping_embedding', 3)
text_augmenter(sample, 'word_swapping_random', 3)
