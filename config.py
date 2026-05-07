# config.py — Global parameters for the NewsProject pipeline

N_SPLITS      = 5
RANDOM_STATE  = 42
SHUFFLE       = True
USE_STOPWORDS = True       # genre + keyword preprocessing
MIN_TOKEN_LEN = 2
BINARIZE      = False
ALPHA         = 1.0
REPORT        = True
BBC_DATA_DIR  = "bbc"
BBC_RSS_URL   = "https://feeds.bbci.co.uk/news/rss.xml"
SAMPLE_N      = 3          # articles shown in detail during demo

BBC_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]
