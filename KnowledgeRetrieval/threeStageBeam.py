from tool import *
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import *

def dataMain():
    with open(KG_PATH, "rb") as f:
        G = pickle.load(f)

    print(f"Number of nodes: {G.vcount()}")
    print(f"Number of edges: {G.ecount()}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)

    datas = load_json(DATA_JSON)
    datas_LLM = []
    num = 0
    for data in tqdm(datas,desc=f"Outer Progress", ncols=100, colour="red"):
        num = num + 1
        # question
        message = data["question"]
        options = data["options"]
        question = message + "\n" + str(options)
        match_entity = []
        LLM_V3_Analysis_KG = data.get('LLM_Analysis_Ner')
        if LLM_V3_Analysis_KG == "[]" or LLM_V3_Analysis_KG == None:
            data["medLTRAGAnswer"] = data["chart5MAnswer"]
            data["medLTRAG"] = ""
            data["depth"] = 0
            datas_LLM.append(data)
            continue
        # entity
        topic_entity_A = LLM_V3_Analysis_KG.get('A')
        if topic_entity_A != None:
            match_entity.extend(topic_entity_A)

        topic_entity_B = LLM_V3_Analysis_KG.get('B')
        if topic_entity_B != None:
            match_entity.extend(topic_entity_B)

        topic_entity_C = LLM_V3_Analysis_KG.get('C')
        if topic_entity_C != None:
            match_entity.extend(topic_entity_C)

        topic_entity_D = LLM_V3_Analysis_KG.get('D')
        if topic_entity_D != None:
            match_entity.extend(topic_entity_D)

        topic_entity_E = LLM_V3_Analysis_KG.get('E')
        if topic_entity_E != None:
            match_entity.extend(topic_entity_E)

        match_entity = list(set(match_entity))

        # match_entity
        topic_entity = match_the_entity(match_entity)
        topic_entity = list(set(topic_entity))
        topic_entity = sort_by_letters(topic_entity)

        cluster_chain_of_entities = []
        chain_of_paths = []
        if len(topic_entity) == 0:
            data["medLTRAGAnswer"] = data["chart5MAnswer"]
            data["medLTRAG"] = ""
            data["depth"] = 0
            datas_LLM.append(data)
            continue
        pre_relations = [""] * len(topic_entity)
        pre_heads = [False] * len(topic_entity)
        flag_printed = False
        for depth in range(1, ARGS_DEPTH + 1):
            current_entity_relations_list = []
            i = 0
            for entity in topic_entity:
                # Find edges related to entities + pruning
                retrieve_relations_with_scores = relation_search_prune(tokenizer,model,question, entity, pre_relations[i], pre_heads[i], G)
                current_entity_relations_list.extend(retrieve_relations_with_scores)

                i += 1
            total_candidates = []
            total_relations = []
            total_topic_entities = []
            total_head = []
            # Search for the next entity in the entity-relation set
            for entity in current_entity_relations_list:
                if entity['head'] == True:
                    # Find the corresponding entities in the entity-relation set + pruning
                    entity_candidates_name = entity_search(tokenizer,model,question,entity['entity'], entity['relation'],G, True)
                else:
                    entity_candidates_name = entity_search(tokenizer,model,question,entity['entity'], entity['relation'],G, False)


                if len(entity_candidates_name) == 0:
                    continue
                total_candidates, total_relations, total_topic_entities, total_head = update_history(
                    entity_candidates_name, entity, total_candidates,
                    total_relations, total_topic_entities, total_head)

            if len(total_candidates) == 0:
                data["medLTRAGAnswer"] = data["chart5MAnswer"]
                data["medLTRAG"] = ""
                data["depth"] = 0
                datas_LLM.append(data)
                flag_printed = True
                break
            #  Relations and entities, overall reranking + pruning
            chain_of_entities = paths_prune_agin(tokenizer,model,question, total_relations, total_candidates, total_topic_entities, total_head, depth)
            cluster_chain_of_path,chain_of_path = paths_package(cluster_chain_of_entities,chain_of_entities,depth)

            cluster_chain_of_entities=cluster_chain_of_path
            chain_of_paths = chain_of_path

            if len(chain_of_paths) == 0:
                data["medLTRAGAnswer"] = data["chart5MAnswer"]
                data["medLTRAG"] = ""
                data["depth"] = 0
                datas_LLM.append(data)
                flag_printed = True
                break

            stop, results = reasoning(question, chain_of_paths)

            if stop:
                print("============= stoped at depth %d." % depth)
                data["chain_of_paths"] = chain_of_paths
                data["medLTRAGAnswer"] = results
                data["medLTRAG"] = ""
                data["depth"] = depth
                datas_LLM.append(data)
                flag_printed = True
                break
            else:
                match_entity = []
                relation = []
                head = []
                for entity in chain_of_entities:
                    if 'match_kg' in entity:
                        match_entity.extend(entity["match_kg"])
                        for index in range(len(entity["match_kg"])):
                            relation.append(entity["relation"])
                            head.append(entity["head"])
                pre_relations = relation
                pre_heads = head
                topic_entity = match_entity
                continue

        if not flag_printed:
            data["medLTRAGAnswer"] = data["chart5MAnswer"]
            data["medLTRAG"] = ""
            data["depth"] = 0
            datas_LLM.append(data)

        if num % 10 == 0:
            save_json_add(datas_LLM, FILE_NAME)
            datas_LLM = []

    save_json_add(datas_LLM, FILE_NAME)


if __name__ == '__main__':
     dataMain()

