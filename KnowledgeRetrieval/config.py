# =====================
# path
# =====================
KG_PATH = "../Leprosy/KG_Leprosy_igraph.pkl"
ENTITY_EMBEDDINGS_PATH = "../Leprosy/entity_Leprosy_embeddings.pkl"
KEYWORD_EMBEDDINGS_PATH = "../Leprosy/keyword_Leprosy_embeddings.pkl"

LOCAL_MODEL_PATH = "../MedCPT"

DATA_JSON = "../Leprosy/draft/Leprosy.json"

FILE_NAME = "../Leprosy/draft/Answer_15_8_text.json"

# =====================
# Search parameters
# =====================
MAX_KG = 15             # maximum number of KG expansions per step
MAX_RELATION = 8        # maximum number of relations per entity
MAX_ENTITY = 8          # maximum number of entities per relation

ARGS_DEPTH = 6          # maximum search depth
MAX_NUM = 200           # maximum number of samples
THRESHOLD = 0.93        # matching threshold
BATCH_SIZE = 20         # batch size
DEVICE = "cpu"          # or "cpu"
MODELNAME = "gpt-5-mini-2025-08-07"

# =====================
# Filter Data
# =====================
USELESS_RELATION_LIST = [
    "resembles","synonyms","is a","is_a_form_of","is a form of", "is a type of","is_a_type_of",
    "is_form_of","is form of","subtype","subtype_of","is_subtype_of", "is_a"
]

USELESS_ENTITY_LIST = ["patient","patients","people"]