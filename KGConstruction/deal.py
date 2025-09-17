from Unit import *
from igraph import Graph
import pickle
from sentence_transformers import SentenceTransformer

def datas():
    kgs = []
    for i in range(10):
        filename = f"../Tuberculosis/byYear/PMID_output_{i}.jsonl"
        datas = load_to_jsonl(filename)
        for data in datas:
            out_answer = data["response"]["body"]["choices"][0]["message"]["content"]
            out_answerList = changeStrToList(out_answer)
            if out_answerList:
                kgs.extend(out_answerList)
    with open("../Tuberculosis/byYear/kgs.txt", "w", encoding="utf-8") as f:
        for item in kgs:
            if isinstance(item, tuple) and len(item) == 3:
                f.write(str(item) + "\n")

def creatG():
    edge_dict = {}
    with open('../Tuberculosis/byYear/kgs.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                triple = ast.literal_eval(line)
                if isinstance(triple, tuple):
                    head, relation, tail = triple
                    edge_dict[(head, tail)] = {"relation": relation, "hops": 1}

    node_set = set()
    for h, t in edge_dict.keys():
        node_set.update([h, t])

    node_list = list(node_set)
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    edges_idx = [(node_to_index[h], node_to_index[t]) for h, t in edge_dict.keys()]
    relations = [v["relation"] for v in edge_dict.values()]
    hops_list = [v["hops"] for v in edge_dict.values()]

    g = Graph(directed=True)
    g.add_vertices(len(node_list))
    g.add_edges(edges_idx)

    g.vs['name'] = node_list
    g.es['relation'] = relations
    g.es['hops'] = hops_list

    with open("../Tuberculosis/KG_Tuberculosis_igraph.pkl", "wb") as f:
        pickle.dump(g, f)

    with open("../Tuberculosis/KG_Tuberculosis_igraph.pkl", "rb") as f:
        g_loaded = pickle.load(f)
        print(f"Number of nodes:{g_loaded.vcount()}")
        print(f"Number of edges: {g_loaded.ecount()}")


def EmbeddingsKG():
    # entities
    entity = set()
    with open('../Tuberculosis/byYear/kgs.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                triple = ast.literal_eval(line)
                if isinstance(triple, tuple):
                    head, relation, tail = triple
                    entity.add(head)
                    entity.add(tail)

    entity = list(entity)
    # keywords
    keyword = set()
    dataKG = load_json("../Tuberculosis/draft/Tuberculosis.json")
    for item in dataKG:
        LLM_V3_Analysis_KG = item.get("LLM_Analysis_Ner")
        if LLM_V3_Analysis_KG == "[]" or LLM_V3_Analysis_KG == None:
            continue
        KG_A = LLM_V3_Analysis_KG.get("A")
        if KG_A != None:
            for item_kg in KG_A:
                keyword.add(item_kg)
        KG_B = LLM_V3_Analysis_KG.get("B")
        if KG_B != None:
            for item_kg in KG_B:
                keyword.add(item_kg)
        KG_C = LLM_V3_Analysis_KG.get("C")
        if KG_C != None:
            for item_kg in KG_C:
                keyword.add(item_kg)
        KG_D = LLM_V3_Analysis_KG.get("D")
        if KG_D != None:
            for item_kg in KG_D:
                keyword.add(item_kg)
        KG_E = LLM_V3_Analysis_KG.get("E")
        if KG_E != None:
            for item_kg in KG_E:
                keyword.add(item_kg)

    keyword = list(keyword)
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # encode entities
    embeddings = model.encode(entity, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
    entity_emb_dict = {
        "entities": entity,
        "embeddings": embeddings,
    }
    with open("../Tuberculosis/entity_Tuberculosis_embeddings.pkl", "wb") as f:
        pickle.dump(entity_emb_dict, f)
    # encode keywords
    embeddings = model.encode(keyword, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
    keyword_emb_dict = {
        "keywords": keyword,
        "embeddings": embeddings,
    }
    with open("../Tuberculosis/keyword_Tuberculosis_embeddings.pkl", "wb") as f:
        pickle.dump(keyword_emb_dict, f)

if __name__ == '__main__':
    # datas()
    # creatG()
    EmbeddingsKG()