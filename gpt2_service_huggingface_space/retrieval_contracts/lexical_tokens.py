# The one word splitter for lexical (BM25) search, used when the index is built and when a question is searched;
# any change here is a new LEXICAL_TOKENIZER_VERSION and a rebuilt index, because both sides must split words identically
import re  # word pattern
import unicodedata  # folds look-alike characters (ligatures, full-width letters)

LEXICAL_TOKENIZER_VERSION = 1  # stored in the bundle manifest; the service refuses a bundle built with another version
WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")  # runs of letters and digits, inner apostrophes kept for the possessive step
CURLY_APOSTROPHES = {"’": "'", "‘": "'", "ʼ": "'"}  # the books use curly quotes, questions usually straight ones
STOPWORDS = frozenset({  # function and question words: they carry no evidence and would flood the postings
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been",
    "before", "being", "but", "by", "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has", "have",
    "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "me",
    "might", "more", "most", "much", "must", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "out", "over", "own", "said", "same", "say", "says", "shall", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "upon", "us", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "whose", "why", "will", "with", "would", "yes", "yet", "you", "your", "yours",
})


def normalize_word(word: str) -> str:  # possessive and plural endings off, so "Lannisters'" and "Lannister's" match "lannister"
    if word.endswith("'s"):  # possessive "Tyrion's"
        word = word[:-2]  # becomes "tyrion"
    word = word.replace("'", "")  # remaining apostrophes go ("don't" -> "dont")
    if len(word) > 4 and word.endswith("ies"):  # "ladies" -> "lady"
        return word[:-3] + "y"  # singular "-y" form
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):  # plain plural "-s"
        return word[:-1]  # "starks" -> "stark", while "dragonglass" and "maesterius" stay whole
    return word  # already singular or too short to touch


def split_lexical_tokens(text: str) -> list[str]:  # the lexical tokens of one text, in order, repeats kept (term counts need them)
    folded_text = unicodedata.normalize("NFKC", text).lower()  # one form per character, lower case
    for curly_apostrophe, straight_apostrophe in CURLY_APOSTROPHES.items():  # every curly apostrophe kind
        folded_text = folded_text.replace(curly_apostrophe, straight_apostrophe)  # made straight
    tokens = []  # kept tokens
    for word in WORD_PATTERN.findall(folded_text):  # every word in reading order
        base_word = word[:-2] if word.endswith("'s") else word  # stopword check ignores a possessive ending
        if base_word.replace("'", "") in STOPWORDS:  # function word
            continue  # dropped before stemming, so "does" never becomes "doe"
        tokens.append(normalize_word(word))  # the searchable form
    return [token for token in tokens if token]  # an apostrophe-only word can end up empty
