import json
import random

animals = [
    "dog", "cat", "elephant", "tiger", "lion", "giraffe", "zebra", "kangaroo",
    "panda", "polar bear", "wolf", "fox", "rabbit", "mouse", "horse", "eagle",
    "shark", "whale", "dolphin", "penguin", "koala", "leopard", "cheetah",
    "bear", "monkey", "gorilla", "snake", "crocodile"
]

not_animals = [
    "car", "table", "computer", "house", "tree", "phone", "book", "chair",
    "pencil", "river", "mountain", "cloud", "building", "train", "bicycle",
    "bus", "laptop", "shoe", "bag", "window", "door", "road", "rock"
]

adjectives = [
    "big", "small", "huge", "tiny", "scary", "cute", "wild", "white", "black",
    "angry", "fast", "lazy", "beautiful", "dangerous", "rare", "young", "old",
    "brown", "spotted", "striped", "furry", "wet", "sleeping", "running"
]

templates = [
    "There is a {adj} {obj} in the picture.",
    "Look at this {adj} {obj}.",
    "I can see a {obj} over there.",
    "Is that a {obj} on the grass?",
    "A {adj} {obj} is sitting in the corner.",
    "The photo shows a {obj}.",
    "This {obj} looks {adj}.",
    "Look, a {obj}!",
    "I think this is a {obj}.",

    "The {adj} {obj} is running very fast.",
    "I saw a {obj} jumping over the fence.",
    "Watch out, the {obj} is hunting!",
    "A sleeping {obj} lies under the tree.",
    "The {obj} is eating its food.",
    "Can you capture the {obj} on camera?",
    "The {obj} swims in the water.",
    "We watched the {obj} playing with a ball.",

    "Behind the bush, a {obj} is hiding.",
    "On the left side, there is a large {obj}.",
    "In the background, you can spot a {obj}.",
    "Next to the river stands a {obj}.",
    "Above the rock, a {obj} is watching us.",
    "Deep in the forest lives a {obj}.",
    "Right in front of me is a {obj}.",

    "Could this be a {obj}?",
    "What is that {adj} {obj} doing here?",
    "Do you think the {obj} is friendly?",
    "Have you ever seen a real {obj}?",
    "Is it safe to go near that {obj}?",
    "Why is the {obj} making that sound?",
    "Who owns this {obj}?",

    "What a majestic {obj}!",
    "I am afraid of that {obj}.",
    "This is the most beautiful {obj} I have ever seen.",
    "That {obj} has amazing colors.",
    "The {obj} seems very calm today.",
    "I wish I had a pet {obj}.",
    "The fur of the {obj} is very soft.",
    "Check out the size of that {obj}."
]

def tokenize_and_tag(text, target_word, tag_type):
    """
    Splits the text into tokens and assigns BIO tags.
    If target_word is None, all tags will be 'O'
    """
    for punct in [".", ",", "!", "?", ":"]:
        text = text.replace(punct, f" {punct}")
    tokens = text.split()
    tags = ["O"] * len(tokens)
    if target_word is None:
        return tokens, tags
    target_tokens = target_word.split()
    target_len = len(target_tokens)
    for i in range(len(tokens) - target_len + 1):
        current_slice = tokens[i: i + target_len]
        if [t.lower() for t in current_slice] == [t.lower() for t in target_tokens]:
            tags[i] = f"B-{tag_type}"
            for j in range(1, target_len):
                tags[i + j] = f"I-{tag_type}"
            break

    return tokens, tags


def generate_dataset(total_samples=1000):
    data = []

    for _ in range(total_samples):
        template = random.choice(templates)
        adj = random.choice(adjectives) if random.random() > 0.3 else ""
        is_positive = random.random() > 0.3

        if is_positive:
            obj = random.choice(animals)
            label = "ANIMAL"
            target = obj
        else:
            obj = random.choice(not_animals)
            label = "O"
            target = None
        phrase = template.format(adj=adj, obj=obj)
        phrase = " ".join(phrase.split())
        if random.random() > 0.9:
            phrase = phrase.upper()
            if target: target = target.upper()

        tokens, tags = tokenize_and_tag(phrase, target, "ANIMAL")

        data.append({
            "tokens": tokens,
            "ner_tags": tags
        })

    return data

if __name__ == "__main__":
    dataset = generate_dataset(1200)
    save_path = 'data.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Samples sucessuflly generated: {len(dataset)}")
    print(f"File saved as: {save_path}")