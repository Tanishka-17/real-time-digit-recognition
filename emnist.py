import tensorflow_datasets as tfds

ds, info = tfds.load(
    "emnist/letters",
    split="train",
    with_info=True
)

print(info)