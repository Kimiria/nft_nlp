"""
pip install bertopic
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic

# Data preparation

df = pd.read_csv('./data/captions_dataset.csv')
df.path = df.path.str.replace("./dataset/image/", "")
df.path = df.path.str.replace("./dataset/gif/", "")
df.path = df.path.str.replace("./dataset/video/", "")

captions = pd.read_csv('./data/captions.csv')
captions.drop(['Unnamed: 0'], axis=1, inplace=True)
captions.caption = captions.caption.str.replace("<start>", "")
captions.caption = captions.caption.str.replace("<end>", "")
captions.caption = captions.caption.str.replace(".", "")
captions = captions.merge(df, how='left',
                          left_on='image_name', right_on='path')
docs = list(captions.caption.values)

# Model training
topic_model = BERTopic(min_topic_size=15, language="english",
                       calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(docs)

# Extracting frequent topics
freq = topic_model.get_topic_info()
freq.head(5)
topic_model.get_topic(0)


# Visualization

topic_model.visualize_topics()
topic_model.visualize_distribution(probs[0], min_probability=0.0015)
topic_model.visualize_hierarchy(top_n_topics=50)
topic_model.visualize_barchart(top_n_topics=5)
topic_model.visualize_heatmap(n_clusters=20, width=1000, height=1000)
topic_model.visualize_term_rank()

# Updating ans selecting topics viewed before

topic_model.update_topics(docs, topics, n_gram_range=(1, 2))
topic_model.get_topic(0)   # We select topic that we viewed before
new_topics, new_probs = topic_model.reduce_topics(docs, topics,
                                                  probs, nr_topics=60)


# Save model
topic_model.save("my_model")

# Load model
my_model = BERTopic.load("my_model")
