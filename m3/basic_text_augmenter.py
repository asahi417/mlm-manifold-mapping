""" Basic Text Augmentation """
from typing import List

from textattack import augmentation


def text_augmenter(samples: List or str,
                   augment_type: str = 'back_translation',
                   transformations_per_example: int = 1):
    """ text augmentation

    Parameters
    ----------
    samples: List or str
        sentence(s) to transform
    augment_type: str
        type of augmentation
    transformations_per_example: int
        number of examples to augment from one input

    Returns
    -------
    a nested list where each list corresponds to the augmented sentences for original sentence
    """
    assert type(samples) is str or list, f'invalid input type: {samples} ({type(samples)})'
    if augment_type == 'back_translation':
        aug = augmentation.BackTranslationAugmenter(transformations_per_example=transformations_per_example)
    elif augment_type == 'eda':
        aug = augmentation.EasyDataAugmenter(transformations_per_example=transformations_per_example)
    elif augment_type == 'word_swapping_synonym':
        aug = augmentation.WordNetAugmenter(transformations_per_example=transformations_per_example)
    elif augment_type == 'word_swapping_embedding':
        aug = augmentation.EmbeddingAugmenter(transformations_per_example=transformations_per_example)
    elif augment_type == 'word_swapping_random':
        aug = augmentation.recipes.SwapAugmenter(transformations_per_example=transformations_per_example)
    else:
        raise ValueError(f'unknown method: {augment_type}')
    return aug.augment(samples)

